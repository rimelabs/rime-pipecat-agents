import argparse
import datetime
import io
import logging
import os
import wave
from typing import Dict, Callable

import aiofiles
from dotenv import load_dotenv

from pipecat.frames.frames import EndFrame, TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
from pipecat.services.rime.tts import RimeTTSService
from pipecat.transcriptions.language import Language
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.network.fastapi_websocket import FastAPIWebsocketParams
from pipecat.transports.services.daily import DailyParams
from pipecat.processors.frameworks.rtvi import RTVIProcessor, RTVIObserver

# Configure logging
logger = logging.getLogger("rime-pipecat")
logger.setLevel(logging.INFO)

# Load environment variables
load_dotenv(override=True)

# Transport configuration mapping
# Each transport is defined as a lambda to avoid premature instantiation
transport_params: Dict[str, Callable[[], TransportParams]] = {
    "daily": lambda: DailyParams(audio_out_enabled=True),
    "twilio": lambda: FastAPIWebsocketParams(audio_out_enabled=True),
    "webrtc": lambda: TransportParams(audio_out_enabled=True, audio_in_enabled=True),
}


async def save_audio_file(
    audio: bytes, filename: str, sample_rate: int, num_channels: int
) -> None:
    """
    Save audio data to a WAV file.

    Args:
        audio: Raw audio data bytes
        filename: Path where the WAV file will be saved
        sample_rate: Audio sample rate in Hz
        num_channels: Number of audio channels (1 for mono, 2 for stereo)
    """
    if not audio:
        logger.warning("No audio data to save for %s", filename)
        return

    try:
        with io.BytesIO() as buffer:
            with wave.open(buffer, "wb") as wf:
                wf.setnchannels(num_channels)  # Set number of channels
                wf.setsampwidth(2)  # Set sample width to 2 bytes (16 bits)
                wf.setframerate(sample_rate)  # Set frame rate
                wf.writeframes(audio)  # Write audio data

            # Save the buffer contents to file
            async with aiofiles.open(filename, "wb") as file:
                await file.write(buffer.getvalue())

        logger.info("Audio successfully saved to %s", filename)
    except Exception as e:
        logger.error("Failed to save audio to %s: %s", filename, str(e))


async def run_example(
    transport: BaseTransport, args: argparse.Namespace, handle_sigint: bool
) -> None:
    """
    Run the Rime TTS bot example.

    This function sets up and runs a pipeline that:
    1. Initializes the Rime TTS service
    2. Sets up audio buffering
    3. Responds to client connections with predefined TTS messages
    4. Records the bot's audio output if recording is enabled

    Args:
        transport: The transport layer to use (Daily, Twilio, or WebRTC)
        args: Command line arguments containing record flag and text content
        handle_sigint: Whether to handle interrupt signals
    """
    try:
        logger.info("Starting Rime TTS bot example")

        # Initialize RTVI processor
        rtvi = RTVIProcessor()

        # Initialize Rime TTS service
        api_key = os.getenv("RIME_API_KEY")
        if not api_key:
            raise ValueError("RIME_API_KEY environment variable not set")

        logger.info("Initializing Rime TTS service")
        tts = RimeTTSService(
            api_key=api_key,
            voice_id="rex",
            model="mistv2",
            url="wss://users.rime.ai/ws2",
            params=RimeTTSService.InputParams(
                language=Language.EN,
                speed_alpha=1.0,
                reduce_latency=False,
                pause_between_brackets=True,
                phonemize_between_brackets=False,
            ),
        )

        # Default text
        default_text = (
            "Welcome! This is a demonstration of Rime's Text-to-Speech capabilities. "
            "The voice you're hearing is generated in real-time using advanced AI technology."
        )

        # Use text from file if available
        text_to_use = args.textfile if args.textfile is not None else default_text

        # Initialize audio buffer for recording
        audiobuffer = AudioBufferProcessor()

        # Set up the pipeline
        # Pipeline is the actual chain of frame processors (like TTS, LLM, STT services) connected in sequence
        # PipelineTask is the central orchestrator that manages pipeline execution, frame routing, and lifecycle events
        pipeline_params = PipelineParams(
            enable_metrics=True, enable_usage_metrics=True)

        task = PipelineTask(
            Pipeline([transport.input(), rtvi, tts,
                     transport.output(), audiobuffer]),
            params=pipeline_params,
        )
        rtvi_observer = RTVIObserver(rtvi)
        task.add_observer(rtvi_observer)

        # Handle client connection events
        # The queue_frames() method allows you to inject frames into the pipeline for processing
        @transport.event_handler("on_client_connected")
        async def on_client_connected(transport, client) -> None:
            """Handle new client connections by starting recording and sending welcome messages."""
            if args.record:
                await audiobuffer.start_recording()
            await task.queue_frames([
                TTSSpeakFrame(text_to_use),
                EndFrame(),
            ])

        # Handle audio recording - Handler for separate tracks
        @audiobuffer.event_handler("on_track_audio_data")
        async def on_track_audio_data(
            buffer,
            user_audio,
            bot_audio,
            sample_rate,
            num_channels,
        ) -> None:
            """Save bot's audio output to a WAV file."""
            if not args.record:
                return

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs("recordings", exist_ok=True)

            # Save bot audio
            bot_filename = f"recordings/bot_{timestamp}.wav"
            await save_audio_file(bot_audio, bot_filename, sample_rate, 1)

        # PipelineRunner is the high-level execution manager that runs pipeline tasks
        # with lifecycle and signal handling. The handle_sigint parameter controls whether
        # PipelineRunner automatically handles system interrupt signals (SIGINT and SIGTERM)
        # for graceful shutdown and resource cleanup
        runner = PipelineRunner(handle_sigint=handle_sigint)
        await runner.run(task)
    except Exception as e:
        logger.error("Error in run_example: %s", str(e))
        raise  # Re-raise the exception to ensure the caller knows about the failure


if __name__ == "__main__":
    # Import standard utility for running example bot scripts in the Pipecat framework
    from pipecat.examples.run import main

    def validate_text_file(filepath):
        if not filepath:
            return None
        if not os.path.isfile(filepath):
            raise ValueError(f"File not found: {filepath}")
        if not filepath.endswith('.txt'):
            raise ValueError(f"File must be a .txt file: {filepath}")
        with open(filepath, 'r') as file:
            return file.read().strip()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--record", action="store_true", default=False, help="Enable audio recording"
    )
    parser.add_argument(
        "--textfile", type=validate_text_file, help="Path to a text file to replace the default text"
    )

    logger.info("Starting the bot")

    # Pipecat Examples Runner Utility
    # -----------------------------
    #
    # A standardized utility for running example bot scripts in the Pipecat framework. This utility
    # enables developers to build and test their bots using consistent patterns across different
    # transport layers.
    #
    # Usage:
    #     The main function accepts two parameters:
    #     1. run_example: Your bot's main execution function
    #     2. transport_params: A dictionary defining available transports:
    #        - "daily": Daily.co WebRTC
    #        - "twilio": Twilio
    #        - "webrtc": Direct WebRTC
    #
    # Key Benefits:
    #     - Transport Agnostic: Write bot logic once, run it with different transports
    #     - Flexible Testing: Switch between transport layers via command-line arguments
    #     - Standardized Pattern: Follows Pipecat's foundational example structure
    #
    # Note:
    #     This utility is primarily intended for local development and testing. Use it to
    #     prototype and validate your Pipecat bots before setting up production infrastructure.
    main(run_example, transport_params=transport_params, parser=parser)
