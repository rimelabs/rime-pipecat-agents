import argparse
import asyncio
import datetime
import io
import logging
import os
import wave
from typing import Optional

import aiofiles
from dotenv import load_dotenv

from pipecat.frames.frames import EndFrame, TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
from pipecat.services.rime.tts import RimeTTSService
from pipecat.transcriptions.language import Language
from pipecat.processors.frameworks.rtvi import RTVIProcessor, RTVIObserver
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams

# Configure logging
logger = logging.getLogger("rime-pipecat")
logger.setLevel(logging.INFO)

# Load environment variables
load_dotenv(override=True)


async def save_audio_file(
    audio: bytes, filename: str, sample_rate: int, num_channels: int
) -> None:
    """Save audio data to a WAV file."""
    if not audio:
        logger.warning("No audio data to save for %s", filename)
        return

    try:
        with io.BytesIO() as buffer:
            with wave.open(buffer, "wb") as wf:
                wf.setnchannels(num_channels)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(audio)

            async with aiofiles.open(filename, "wb") as file:
                await file.write(buffer.getvalue())

        logger.info("Audio successfully saved to %s", filename)
    except Exception as e:
        logger.error("Failed to save audio to %s: %s", filename, str(e))


async def run_console_tts(args: argparse.Namespace) -> None:
    """Run TTS in console mode with local audio output."""
    try:
        logger.info("Starting console TTS mode")

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

        # Initialize RTVI processor
        rtvi = RTVIProcessor()

        # Default text
        default_text = (
            "There's a 2022 Ferrari F8 Tributo with 7,638 miles, a 2018 Ferrari 488 G. T. B. with 9,837 miles, and a 2019 Ferrari G. T. C. 4 Lusso V12 with 17,097 miles."
        )

        # Determine which text to use
        text_to_use = args.text if args.text is not None else (
            args.textfile if args.textfile is not None else default_text)

        # Initialize audio buffer for recording
        audiobuffer = AudioBufferProcessor()

        # Create local audio transport
        transport = LocalAudioTransport(LocalAudioTransportParams(
            audio_out_enabled=True,
            audio_in_enabled=False,  # We don't need input for TTS-only
        ))

        # Set up the pipeline
        pipeline_params = PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True
        )

        task = PipelineTask(
            Pipeline([transport.input(), tts, rtvi,
                     transport.output(), audiobuffer]),
            params=pipeline_params,
            enable_tracing=True,
            enable_turn_tracking=True,
        )

        # Handle audio recording
        @audiobuffer.event_handler("on_track_audio_data")
        async def on_track_audio_data(
            buffer, user_audio, bot_audio, sample_rate, num_channels,
        ) -> None:
            """Save bot's audio output to a WAV file."""
            if not args.record:
                return

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs("recordings", exist_ok=True)

            bot_filename = f"recordings/bot_{timestamp}.wav"
            await save_audio_file(bot_audio, bot_filename, sample_rate, 1)

        # Use task start event to trigger TTS immediately
        @task.event_handler("on_pipeline_started")
        async def on_pipeline_started(task):
            """Start TTS immediately when task starts."""
            logger.info("Task started, beginning TTS playback")

            if args.record:
                await audiobuffer.start_recording()

            rtvi_observer = RTVIObserver(rtvi)
            task.add_observer(rtvi_observer)

            # Queue the TTS frames
            await task.queue_frames([
                TTSSpeakFrame(text_to_use),
                EndFrame(),
            ])

        # Run the pipeline
        runner = PipelineRunner(handle_sigint=True)
        await runner.run(task)

    except Exception as e:
        logger.error("Error in console TTS: %s", str(e))
        raise


def validate_text_file(filepath: Optional[str]) -> Optional[str]:
    """Validate and read text file."""
    if not filepath:
        return None
    if not os.path.isfile(filepath):
        raise ValueError(f"File not found: {filepath}")
    if not filepath.endswith(".txt"):
        raise ValueError(f"File must be a .txt file: {filepath}")
    with open(filepath, "r") as file:
        return file.read().strip()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Console TTS with Rime and local audio")
    parser.add_argument(
        "--record",
        action="store_true",
        default=False,
        help="Enable audio recording"
    )
    parser.add_argument(
        "--textfile",
        type=validate_text_file,
        help="Path to a text file to replace the default text",
    )
    parser.add_argument(
        "--text",
        type=str,
        help="Direct text input to be spoken by the bot",
    )

    args = parser.parse_args()

    logger.info("Starting console TTS bot")

    # Run the console TTS
    asyncio.run(run_console_tts(args))


if __name__ == "__main__":
    main()
