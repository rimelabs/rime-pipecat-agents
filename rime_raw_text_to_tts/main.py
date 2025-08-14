import argparse
import asyncio
import datetime
import io
import logging
import os
import wave
from typing import Dict, Callable

import aiofiles
from dotenv import load_dotenv
import simpleaudio as sa

from pipecat.frames.frames import Frame, EndFrame, TTSSpeakFrame, TextFrame
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
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


# Configure logging
logger = logging.getLogger("rime-pipecat")
logger.setLevel(logging.INFO)

# Load environment variables
load_dotenv(override=True)

RIME_VOICE_ID = "glacier"
RIME_MODEL = "mistv2"
RIME_URL = "wss://users.rime.ai/ws2"

RIME_API_KEY = os.getenv("RIME_API_KEY")


class SimpleAudioPlayer(AudioBufferProcessor):
    """Collects audio frames and plays them using simpleaudio"""

    def __init__(self):
        super().__init__()
        self.audio_buffer = bytearray()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        # First, let the parent class handle the frame normally
        await super().process_frame(frame, direction)

        # Collect audio frames
        if hasattr(frame, 'audio') and frame.audio:
            self.audio_buffer.extend(frame.audio)

        # Play audio when TTS synthesis stops
        if frame.__class__.__name__ == 'TTSStoppedFrame':
            print("TTS synthesis completed, playing audio...")
            self.play_audio()
            # Add a small delay to ensure audio starts playing
            await asyncio.sleep(0.1)

    def play_audio(self):
        """Play collected audio using simpleaudio"""
        if not self.audio_buffer:
            print("No audio data to play")
            return

        try:
            # Create a wave object from the audio buffer
            wave_obj = sa.WaveObject(
                audio_data=bytes(self.audio_buffer),
                num_channels=self.num_channels,
                bytes_per_sample=2,  # 16-bit audio
                sample_rate=self.sample_rate
            )

            # Play the audio without waiting (non-blocking)
            play_obj = wave_obj.play()

            # Store the play object so it doesn't get garbage collected
            self._current_play_obj = play_obj

            print("Audio playback started!")

        except Exception as e:
            print(f"Error playing audio: {e}")

        # Clear the buffer for next use
        self.audio_buffer.clear()

async def text_to_speech_play(text: str, voice_id: str = None):
    """Convert text to speech and play directly through speakers"""

    if not RIME_API_KEY:
        raise ValueError("RIME_API_KEY environment variable not set")

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

    # Audio player for direct playback
    audio_player = SimpleAudioPlayer()

    # Create pipeline
    pipeline = Pipeline([
        tts,
        audio_player
    ])

    # Create task
    task = PipelineTask(pipeline)

    # Create runner with timeout
    runner = PipelineRunner()

    # Queue the text and end frame
    await task.queue_frames([
        TextFrame(text),
        EndFrame()
    ])

    # Run the pipeline with a timeout
    try:
        # Wait for the pipeline to complete or timeout after 10 seconds
        await asyncio.wait_for(runner.run(task), timeout=10.0)
    except asyncio.TimeoutError:
        print("Pipeline timeout reached, exiting...")
        await runner.cancel()
    except Exception as e:
        print(f"Pipeline error: {e}")
        await runner.cancel()

def main():
    parser = argparse.ArgumentParser(description="Convert text to speech and play through speakers")
    parser.add_argument("text", help="Text to convert to speech")
    parser.add_argument("-v", "--voice", help="Voice ID to use")

    args = parser.parse_args()


    # Run the conversion and playback
    asyncio.run(text_to_speech_play(args.text, args.voice))
    return 0

if __name__ == "__main__":
    exit(main())