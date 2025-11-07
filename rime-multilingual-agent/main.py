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
    "eng": {"speakerId": "andromeda", "modelId": "arcana", "lang": "eng"},
    "spa": {"speakerId": "sirius", "modelId": "arcana", "lang": "spa"},
    "fra": {"speakerId": "destin", "modelId": "arcana", "lang": "fra"},
    "ger": {"speakerId": "klaus", "modelId": "mistv2", "lang": "ger"},
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


class LanguageDetectionProcessor(FrameProcessor):
    """Processor that detects language from transcription and updates TTS settings."""

    def __init__(self, api_key: str, shared_state: SharedState):
        super().__init__()
        self._client = AsyncOpenAI(api_key=api_key)
        self._frame_buffer = []
        self._shared_state = shared_state
        self._language_detected = False

    async def _detect_language(self, text: str) -> dict:
        """Make OpenAI API call to detect language.
        
        Returns a dict with 'lang' key containing one of: eng, spa, fra, ger
        """
        response = await self._client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Detect the language of the user's text. Respond with JSON containing a 'lang' key with one of: 'eng' (English), 'spa' (Spanish), 'fra' (French), 'ger' (German).",
                },
                {"role": "user", "content": text},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        logger.info(
            f"Language detection response: {response.choices[0].message.content}"
        )
        return json.loads(response.choices[0].message.content)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames and detect language before passing to LLM.
        
        Buffers transcription frames until user stops speaking, then detects
        language and updates TTS settings accordingly.
        """
        await super().process_frame(frame, direction)

        # Handle UserStoppedSpeakingFrame and EndFrame - triggers language detection
        if isinstance(frame, (UserStoppedSpeakingFrame, EndFrame)):
            if self._frame_buffer:
                # Combine all buffered text
                full_text = " ".join(
                    f.text
                    for f, _ in self._frame_buffer
                    if isinstance(f, TranscriptionFrame)
                )
                
                if full_text.strip():
                    logger.info(f"Detecting language for text: {full_text}")
                    language_result = await self._detect_language(full_text)
                    detected_lang = language_result.get("lang")
                    logger.info(f"Detected language: {detected_lang}")
                    
                    if detected_lang in RIME_LANGUAGE_MAP:
                        self._shared_state.language_detected = detected_lang
                        lang_config = RIME_LANGUAGE_MAP[detected_lang]
                        tts_update_frame = TTSUpdateSettingsFrame(
                            settings={
                                "voice_id": lang_config["speakerId"],
                                "model": lang_config["modelId"],
                                "lang": lang_config["lang"],
                            }
                        )
                        await self.push_frame(tts_update_frame, FrameDirection.DOWNSTREAM)
                        logger.info(f"Updated TTS settings for language: {detected_lang}")

                # Push all buffered frames downstream to LLM
                for buffered_frame, buffered_direction in self._frame_buffer:
                    await self.push_frame(buffered_frame, buffered_direction)
                self._frame_buffer.clear()

            # Push the UserStoppedSpeakingFrame or EndFrame
            await self.push_frame(frame, direction)

        # Pass through system frames immediately
        elif isinstance(frame, SystemFrame):
            await self.push_frame(frame, direction)

        # Buffer transcription frames until user stops speaking
        elif isinstance(frame, TranscriptionFrame):
            self._frame_buffer.append((frame, direction))

        # Pass through all other frames
        else:
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
            LanguageDetectionProcessor(api_key=OPENAI_API_KEY, shared_state=shared_state),
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
