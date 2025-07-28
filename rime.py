import argparse
import logging
import os
import datetime
import wave
import io
import aiofiles

from dotenv import load_dotenv
from pipecat.transports.services.daily import DailyParams
from pipecat.transports.network.fastapi_websocket import FastAPIWebsocketParams
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.services.rime.tts import RimeTTSService
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.pipeline import Pipeline
from pipecat.frames.frames import EndFrame, TTSSpeakFrame
from pipecat.transcriptions.language import Language
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor


# Configure logger
logger = logging.getLogger("rime-pipecat")
logger.setLevel(logging.INFO)


load_dotenv(override=True)


# We store functions so objects (e.g. SileroVADAnalyzer) don't get
# instantiated. The function will be called when the desired transport gets
# selected.
transport_params = {
    "daily": lambda: DailyParams(audio_out_enabled=True),
    "twilio": lambda: FastAPIWebsocketParams(audio_out_enabled=True),
    "webrtc": lambda: TransportParams(audio_out_enabled=True, audio_in_enabled=True),
}


async def save_audio_file(audio: bytes, filename: str, sample_rate: int, num_channels: int):
    """Save audio data to a WAV file."""
    if len(audio) > 0:
        with io.BytesIO() as buffer:
            with wave.open(buffer, "wb") as wf:
                wf.setsampwidth(2)
                wf.setnchannels(num_channels)
                wf.setframerate(sample_rate)
                wf.writeframes(audio)
            async with aiofiles.open(filename, "wb") as file:
                await file.write(buffer.getvalue())
        logger.info(f"Audio saved to {filename}")


async def run_example(transport: BaseTransport, _: argparse.Namespace, handle_sigint: bool):
    logger.info("Starting a bot that says one thing")

    logger.info("Rime API Key: %s", os.getenv("RIME_API_KEY"))
    logger.info("Rime tts initialization")
    tts = RimeTTSService(
        api_key=os.getenv("RIME_API_KEY"),
        voice_id="rex",
        model="mistv2",
        url="wss://users.rime.ai/ws2",
        params=RimeTTSService.InputParams(
            language=Language.EN,
            speed_alpha=1.0,
            reduce_latency=False,
            pause_between_brackets=True,
            phonemize_between_brackets=False

        )
    )
    logger.info("pipeline Setup")
    audiobuffer = AudioBufferProcessor()

    # PipelineTask is the central orchestrator that manages pipeline execution, frame routing, and lifecycle events
    # Pipeline is the actual chain of frame processors (like TTS, LLM, STT services) connected in sequence
    params = PipelineParams(enable_metrics=True,
                            enable_usage_metrics=True)
    task = PipelineTask(Pipeline([
        transport.input(),
        tts,
        transport.output(),
        audiobuffer]), params=params)

    # Register an event handler so we can play the audio when the client joins
    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        # The queue_frames() method allows you to inject frames into the pipeline for processing:
        await audiobuffer.start_recording()
        await task.queue_frames([
            TTSSpeakFrame(
                "Welcome! This is a demonstration of Rime's Text-to-Speech capabilities. The voice you're hearing is generated in real-time using advanced AI technology."),
            TTSSpeakFrame(
                "We are using pipecat to build a bot that can say one thing."),
            EndFrame()])
    # PipelineRunner is the high-level execution manager that runs pipeline tasks with lifecycle and signal handling .
    # The handle_sigint parameter controls whether PipelineRunner automatically handles system interrupt signals (SIGINT and SIGTERM) for graceful shutdown , Resource Cleanup:

    # Handler for separate tracks
    @audiobuffer.event_handler("on_track_audio_data")
    async def on_track_audio_data(buffer, user_audio, bot_audio, sample_rate, num_channels):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("recordings", exist_ok=True)
        # Save bot audio
        bot_filename = f"recordings/bot_{timestamp}.wav"
        await save_audio_file(bot_audio, bot_filename, sample_rate, 1)

    runner = PipelineRunner(handle_sigint=handle_sigint)

    await runner.run(task)


if __name__ == "__main__":
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
    from pipecat.examples.run import main
    logger.info("Starting the bot")
    main(run_example, transport_params=transport_params)
