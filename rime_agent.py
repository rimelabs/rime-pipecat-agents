import argparse
import datetime
import io
import logging
import os
import wave
from typing import Dict, Callable

import aiofiles
from dotenv import load_dotenv

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
from pipecat.services.rime.tts import RimeTTSService
from pipecat.transcriptions.language import Language
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.network.fastapi_websocket import FastAPIWebsocketParams
from pipecat.transports.services.daily import DailyParams
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.processors.frameworks.rtvi import RTVIProcessor, RTVIObserver


# Configure logging
logger = logging.getLogger("rime-pipecat")
logger.setLevel(logging.INFO)

# Load environment variables
load_dotenv(override=True)

RIME_VOICE_ID = "glacier"
RIME_MODEL = "mistv2"
RIME_URL = "wss://users.rime.ai/ws2"

RIME_API_KEY = os.getenv("RIME_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

SYSTEM_PROMPT = """
You are witty, friendly, but professional AI assistant powered by Rime AI, a TTS provider
with the most realistic voices on the market.

Everything you say will be spoken by a tts model.

You are built using the Pipecat framework, which is a powerful tool for building voice agents.
"""

transport_params: Dict[str, Callable[[], TransportParams]] = {
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(),
    )
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
    Run the Rime conversational AI bot example.

    This function sets up and runs a pipeline that:
    1. Initializes the Deepgram STT (Speech-to-Text) service
    2. Sets up OpenAI LLM (Language Model) for conversation
    3. Initializes the Rime TTS (Text-to-Speech) service
    4. Sets up audio buffering for processing
    5. Creates a complete conversational pipeline (STT → LLM → TTS)
    6. Records the bot's audio output if recording is enabled

    Args:
        transport: The transport layer to use (Daily, Twilio, or WebRTC)
        args: Command line arguments containing record flag
        handle_sigint: Whether to handle interrupt signals
    """
    try:
        logger.info("Starting Rime TTS bot example")
        rtvi_processor = RTVIProcessor()
        rtvi_observer = RTVIObserver(rtvi_processor)

        # Initialize Rime TTS service
        if not RIME_API_KEY:
            raise ValueError("RIME_API_KEY environment variable not set")
        if not DEEPGRAM_API_KEY:
            raise ValueError("DEEPGRAM_API_KEY environment variable not set")
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        logger.info("Initializing Deepgram STT service")
        stt = DeepgramSTTService(
            api_key=os.getenv("DEEPGRAM_API_KEY"), audio_passthrough=True
        )

        logger.info("Initializing OpenAI LLM service")
        llm = OpenAILLMService(
            model="gpt-4o",
            api_key=os.getenv("OPENAI_API_KEY"),
            params=OpenAILLMService.InputParams(
                temperature=0.7,
            ),
        )

        context = OpenAILLMContext(
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                }
            ],
        )

        context_aggregator = llm.create_context_aggregator(context)

        logger.info("Initializing Rime TTS service")
        tts = RimeTTSService(
            api_key=RIME_API_KEY,
            voice_id=RIME_VOICE_ID,
            model=RIME_MODEL,
            url=RIME_URL,
            params=RimeTTSService.InputParams(
                language=Language.EN,
                speed_alpha=1.0,
                reduce_latency=False,
                pause_between_brackets=True,
                phonemize_between_brackets=False,
            ),
        )

        # Initialize audio buffer for recording
        audiobuffer = AudioBufferProcessor()

        # Set up the pipeline
        pipeline_params = PipelineParams(
            enable_metrics=True, enable_usage_metrics=True)

        task = PipelineTask(
            Pipeline(
                [
                    transport.input(),
                    stt,
                    context_aggregator.user(),
                    llm,
                    tts,
                    rtvi_processor,  # Add this line
                    transport.output(),
                    audiobuffer,
                    context_aggregator.assistant(),
                ]
            ),
            params=pipeline_params,
            enable_tracing=True,
            enable_turn_tracking=True,
        )

        # Handle client connection events
        @transport.event_handler("on_client_connected")
        async def on_client_connected(transport, client) -> None:
            """Handle new client connections by starting recording and sending welcome messages."""
            if args.record:
                await audiobuffer.start_recording()
            logger.info("Client connected")
            task.add_observer(rtvi_observer)

            # Start conversation - empty prompt to let LLM follow system instructions

        @transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(transport, client):
            logger.info("Client disconnected")
            await task.cancel()

        # Handler for merged audio
        @audiobuffer.event_handler("on_audio_data")
        async def on_audio_data(buffer, audio, sample_rate, num_channels):
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"recordings/merged_{timestamp}.wav"

            logger.info("Saving audio to %s", filename)

            os.makedirs("recordings", exist_ok=True)
            await save_audio_file(audio, filename, sample_rate, num_channels)

        # Run the pipeline
        runner = PipelineRunner(handle_sigint=handle_sigint)
        await runner.run(task)

    except ValueError as ve:
        logger.error("Configuration error: %s", str(ve))
        raise
    except Exception as e:
        logger.error(
            "An unexpected error occurred while running the TTS bot: %s", str(
                e)
        )
        logger.exception("Full traceback:")
        raise


if __name__ == "__main__":
    # Import standard utility for running example bot scripts in the Pipecat framework
    from pipecat.examples.run import main

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--record", action="store_true", default=False, help="Enable audio recording"
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
