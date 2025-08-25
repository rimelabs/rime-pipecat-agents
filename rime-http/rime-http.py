import logging
import os
import argparse
from typing import Dict, Callable
from dotenv import load_dotenv
import aiohttp
from pipecat.services.rime.tts import RimeHttpTTSService
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.processors.frameworks.rtvi import RTVIProcessor, RTVIObserver
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.frames.frames import (
    LLMTextFrame,
    LLMFullResponseStartFrame,
    LLMFullResponseEndFrame,
)


# Configure logging
logger = logging.getLogger("rime-pipecat")
logger.setLevel(logging.INFO)

# Load environment variables
load_dotenv(override=True)

RIME_VOICE_ID = "cove"
RIME_MODEL = "mistv2"
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


class LLMCompleteResponseProcessor(FrameProcessor):
    """Processor that aggregates LLM text chunks into complete responses.

    This processor buffers incoming LLM text frames between start and end markers,
    combining them into a single complete response before forwarding.

    Attributes:
        _collecting (bool): Flag indicating if currently collecting text chunks
        _collected_text (str): Buffer for storing collected text chunks
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._collecting = False
        self._collected_text = ""

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMFullResponseStartFrame):
            self._collecting = True
            self._collected_text = ""
            # Pass the start frame through
            await self.push_frame(frame, direction)
        elif isinstance(frame, LLMTextFrame) and self._collecting:
            # Accumulate text but don't push the frame yet
            self._collected_text += frame.text
        elif isinstance(frame, LLMFullResponseEndFrame):
            # Now push the complete text as a single frame
            logger.info("Collected text: %s", self._collected_text)
            if self._collected_text:
                complete_frame = LLMTextFrame(self._collected_text)
                await self.push_frame(complete_frame, direction)
            # Pass the end frame through
            await self.push_frame(frame, direction)
            self._collecting = False
            self._collected_text = ""
        else:
            # Pass all other frames through unchanged
            await self.push_frame(frame, direction)


async def run_example(
    transport: BaseTransport, args: argparse.Namespace, handle_sigint: bool
) -> None:
    """
    Run the Rime conversational AI bot example.

    This function sets up and runs a pipeline that:
    1. Initializes the Deepgram STT (Speech-to-Text) service
    2. Sets up OpenAI LLM (Language Model) for conversation
    3. Initializes the Rime TTS (Text-to-Speech) service
    4. Creates a complete conversational pipeline (STT → LLM → TTS)

    Args:
        transport: The transport layer to use (Daily, Twilio, or WebRTC)
        args: Command line arguments containing additional configuration
        handle_sigint: Whether to handle interrupt signals
    """
    session = None
    try:
        # Validate API keys first
        if not RIME_API_KEY:
            raise ValueError("RIME_API_KEY environment variable not set")
        if not DEEPGRAM_API_KEY:
            raise ValueError("DEEPGRAM_API_KEY environment variable not set")
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        logger.info("Starting Rime TTS bot example")
        rtvi_processor = RTVIProcessor()
        rtvi_observer = RTVIObserver(rtvi_processor)

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
        logger.info("Initializing Rime HTTP service")
        session = aiohttp.ClientSession()
        llm_complete_processor = LLMCompleteResponseProcessor()

        tts = RimeHttpTTSService(
            api_key=RIME_API_KEY,
            voice_id=RIME_VOICE_ID,
            aiohttp_session=session,
            model=RIME_MODEL,
            aggregate_sentences=False,
        )

        # Set up the pipeline
        pipeline_params = PipelineParams(enable_metrics=True, enable_usage_metrics=True)

        task = PipelineTask(
            Pipeline(
                [
                    transport.input(),
                    stt,
                    context_aggregator.user(),
                    llm,
                    llm_complete_processor,
                    tts,
                    rtvi_processor,  # Add this line
                    transport.output(),
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
            """Handle new client connections."""

            logger.info("Client connected")
            task.add_observer(rtvi_observer)

        @transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(transport, client):
            logger.info("Client disconnected")
            await task.cancel()

        # Run the pipeline
        runner = PipelineRunner(handle_sigint=handle_sigint)
        await runner.run(task)

    except ValueError as ve:
        logger.error("Configuration error: %s", str(ve))
        raise
    except Exception as e:
        logger.error(
            "An unexpected error occurred while running the TTS bot: %s", str(e)
        )
        logger.exception("Full traceback:")
        raise
    finally:
        if session:
            logger.info("Closing aiohttp session")
            await session.close()


if __name__ == "__main__":
    # Import standard utility for running example bot scripts in the Pipecat framework
    from pipecat.examples.run import main

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
    main(run_example, transport_params=transport_params)
