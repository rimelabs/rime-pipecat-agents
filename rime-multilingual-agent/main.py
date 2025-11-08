import json
import os

import aiohttp
from deepgram import LiveOptions
from dotenv import load_dotenv
from loguru import logger
from openai import AsyncOpenAI

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    EndFrame,
    Frame,
    SystemFrame,
    TranscriptionFrame,
    TTSUpdateSettingsFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.frameworks.rtvi import RTVIObserver, RTVIProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.rime.tts import RimeHttpTTSService
from pipecat.transcriptions.language import Language
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat_flows import FlowManager, NodeConfig


load_dotenv(override=True)
RIME_VOICE_ID = "astra"
RIME_MODEL = "arcana"
RIME_URL = "wss://users-ws.rime.ai/ws2"

RIME_API_KEY = os.getenv("RIME_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

RIME_LANGUAGE_MAP = {
    Language.EN: {"speakerId": "andromeda", "modelId": "arcana", "lang": "eng"},
    Language.ES: {"speakerId": "sirius", "modelId": "arcana", "lang": "spa"},
    Language.FR: {"speakerId": "destin", "modelId": "arcana", "lang": "fra"},
    Language.DE: {"speakerId": "klaus", "modelId": "mistv2", "lang": "ger"},
}


class SharedState:
    """Shared state container for the conversation."""

    def __init__(self):
        self.language_detected = "eng"


transport_params = {
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(),
    ),
}


def create_initial_node(shared_state: SharedState) -> NodeConfig:
    """Create the initial conversation node."""
    supported_languages = "English, French, Spanish, or German"
    return {
        "name": "conversation",
        "role_messages": [
            {
                "role": "system",
                "content": f"You are a helpful assistant. Be casual and friendly. You support {supported_languages}. If the user speaks a language other than these, politely inform them that you only support {supported_languages} and end the conversation.",
            }
        ],
        "task_messages": [
            {
                "role": "system",
                "content": f"Have a natural conversation with the user in the language they are speaking in ({shared_state.language_detected}). If they speak a language other than {supported_languages}, politely inform them of the supported languages and end the conversation.",
            }
        ],
        "functions": [],
    }


def create_end_node() -> NodeConfig:
    """Create the final node."""
    return {
        "name": "end",
        "task_messages": [
            {
                "role": "system",
                "content": "Thank them and say goodbye.",
            }
        ],
        "post_actions": [{"type": "end_conversation"}],
    }


class LanguageDetectorProcessor(FrameProcessor):
    """Detects language from Deepgram transcription and updates TTS via TTSUpdateSettingsFrame.

    This processor sits between STT and LLM in the pipeline, intercepting transcription
    frames to detect language changes and dynamically reconfigure the TTS service.
    """

    def __init__(self, shared_state: SharedState):
        """Initialize the language detector.

        Args:
            shared_state: Shared state object to track the currently detected language
                         across the pipeline.
        """
        super().__init__()
        self._shared_state: SharedState = shared_state

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames to detect language changes.

        Args:
            frame: The frame to process (typically TranscriptionFrame from Deepgram STT).
            direction: The direction the frame is traveling in the pipeline.
        """
        # Always call parent's process_frame first
        await super().process_frame(frame, direction)

        # Only process final transcription frames that contain Deepgram's result data
        if isinstance(frame, TranscriptionFrame) and frame.result:
            result = frame.result

            # Navigate Deepgram's result structure to extract language information
            # Structure: result.channel.alternatives[0].languages[0]
            if hasattr(result, "channel") and result.channel.alternatives:
                alternative = result.channel.alternatives[0]

                # Check if Deepgram detected any languages (requires multi-language model)
                if hasattr(alternative, "languages") and alternative.languages:
                    detected_lang = alternative.languages[0]

                    try:
                        # Convert Deepgram's language code to Pipecat's Language enum
                        language = Language(detected_lang)

                        # Only update TTS if the language actually changed
                        if language != self._shared_state.language_detected:
                            logger.info(f"Language changed to {language}")

                            # Look up the Rime TTS configuration for this language
                            lang_config = RIME_LANGUAGE_MAP.get(language)
                            if lang_config:
                                # Push a settings update frame downstream to reconfigure TTS
                                # This frame will be intercepted by the Rime TTS service
                                await self.push_frame(
                                    TTSUpdateSettingsFrame(
                                        settings={
                                            "voice_id": lang_config["speakerId"],
                                            "model": lang_config["modelId"],
                                            "lang": lang_config["lang"],
                                        }
                                    ),
                                    FrameDirection.DOWNSTREAM,  # Send toward TTS service
                                )

                                # Update shared state to track the new language
                                self._shared_state.language_detected = language

                    except (ValueError, KeyError) as e:
                        # Handle unsupported languages or missing config gracefully
                        logger.warning(
                            f"Could not convert language '{detected_lang}': {e}"
                        )

        # Always pass the original frame through to the next processor (LLM)
        # This ensures the transcription continues flowing through the pipeline
        await self.push_frame(frame, direction)


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    """Run the multilingual conversation bot."""
    if not RIME_API_KEY:
        raise ValueError("RIME_API_KEY environment variable not set")
    if not DEEPGRAM_API_KEY:
        raise ValueError("DEEPGRAM_API_KEY environment variable not set")
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    # Create shared state for the conversation
    shared_state = SharedState()

    rtvi_processor = RTVIProcessor()
    rtvi_observer = RTVIObserver(rtvi_processor)
    session = aiohttp.ClientSession()

    stt = DeepgramSTTService(
        api_key=DEEPGRAM_API_KEY,
        audio_passthrough=True,
        live_options=LiveOptions(
            language="multi",
        ),
    )
    tts = RimeHttpTTSService(
        api_key=RIME_API_KEY,
        voice_id=RIME_VOICE_ID,
        aiohttp_session=session,
        model=RIME_MODEL,
        params=RimeHttpTTSService.InputParams(
            language=Language.EN,
        ),
    )
    llm = OpenAILLMService(
        model="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    context = LLMContext()
    context_aggregator = LLMContextAggregatorPair(context)

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            LanguageDetectorProcessor(shared_state=shared_state),
            context_aggregator.user(),
            llm,
            tts,
            rtvi_processor,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        enable_tracing=True,
        enable_turn_tracking=True,
    )

    # Initialize flow manager in dynamic mode
    flow_manager = FlowManager(
        task=task,
        llm=llm,
        context_aggregator=context_aggregator,
        transport=transport,
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(_transport, _client):
        logger.info("Client connected")
        task.add_observer(rtvi_observer)
        await flow_manager.initialize(create_initial_node(shared_state))

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(_transport, _client):
        logger.info("Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)
    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
