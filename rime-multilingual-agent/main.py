import datetime
import io
import os
import wave

import aiofiles
import aiohttp
from deepgram import LiveOptions
from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    Frame,
    TranscriptionFrame,
    TTSUpdateSettingsFrame,
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
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor


load_dotenv(override=True)
RIME_API_KEY = os.getenv("RIME_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

RIME_LANGUAGE_MAP = {
    Language.EN: {"speakerId": "celeste", "modelId": "arcana", "lang": "eng"},
    Language.ES: {"speakerId": "astra", "modelId": "arcana", "lang": "spa"},
    Language.FR: {"speakerId": "destin", "modelId": "arcana", "lang": "fra"},
    Language.DE: {"speakerId": "klaus", "modelId": "mistv2", "lang": "ger"},
}

TRANSPORT_PARAMS = {
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(),
    ),
}


async def prepare_audio_filename(prefix: str = "merged") -> str:
    """
    Prepare a filename for audio recording with timestamp.

    Args:
        prefix: Prefix for the filename (default: "merged")

    Returns:
        str: The prepared filename with path
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"recordings/{prefix}_{timestamp}.wav"
    os.makedirs("recordings", exist_ok=True)
    return filename


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
                wf.setnchannels(num_channels)
                wf.setsampwidth(2)  # 16-bit audio
                wf.setframerate(sample_rate)
                wf.writeframes(audio)

            async with aiofiles.open(filename, "wb") as file:
                await file.write(buffer.getvalue())

        logger.info("Audio successfully saved to %s", filename)
    except Exception as e:
        logger.error("Failed to save audio to %s: %s", filename, str(e))


class ConversationState:
    """Current conversation state. Updated by LanguageDetectorProcessor."""

    def __init__(self):
        self.language = Language.EN


class LanguageDetectorProcessor(FrameProcessor):
    """Detects language from Deepgram transcription and updates TTS via TTSUpdateSettingsFrame.

    This processor sits between STT and LLM in the pipeline, intercepting transcription
    frames to detect language changes and dynamically reconfigure the TTS service.
    """

    def __init__(self, conversation_state: ConversationState):
        """Initialize the language detector.

        Args:
            conversation_state: Conversation state object to track the currently detected language
                               across the pipeline.
        """
        super().__init__()
        self._conversation_state: ConversationState = conversation_state

    async def switch_detected_language(self, detected_lang: str):
        """Switch to a new detected language by updating Rime TTS settings.

        Args:
            detected_lang: The language code detected by Deepgram (e.g., 'en', 'es', 'fr').
        """
        try:
            # Convert Deepgram's language code to Pipecat's Language enum
            language = Language(detected_lang)

            # Only update TTS if the language actually changed
            if language != self._conversation_state.language:
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

                    # Update conversation state to track the new language
                    self._conversation_state.language = language

        except (ValueError, KeyError) as e:
            # Handle unsupported languages or missing config gracefully
            logger.warning(f"Could not convert language '{detected_lang}': {e}")

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
                    await self.switch_detected_language(detected_lang)

        # Always pass the original frame through to the next processor (LLM)
        # This ensures the transcription continues flowing through the pipeline
        await self.push_frame(frame, direction)


def create_unsupported_language_node() -> NodeConfig:
    """Create node for handling unsupported languages."""
    return {
        "name": "unsupported_language",
        "role_messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            }
        ],
        "task_messages": [
            {
                "role": "system",
                "content": "Politely inform the user that you only support English, Spanish, French, and German. Tell them the conversation will end.",
            }
        ],
        "post_actions": [
            {"type": "end_conversation", "text": "Thank you for your time!"}
        ],
        "functions": [],
    }


async def handle_unsupported_language(
    flow_manager: FlowManager, detected_language: str
) -> tuple[dict[str, str], NodeConfig]:
    """Handle unsupported language detection.

    Args:
        flow_manager: The FlowManager instance
        detected_language: The language code that was detected
    """
    logger.warning(f"Unsupported language detected: {detected_language}")

    # Update TTS to English
    print(f"Unsupported language detected: {detected_language}")
    english_config = RIME_LANGUAGE_MAP[Language.EN]
    await flow_manager.task.queue_frame(
        TTSUpdateSettingsFrame(
            settings={
                "voice_id": english_config["speakerId"],
                "model": english_config["modelId"],
                "lang": english_config["lang"],
            }
        )
    )

    # Return result and next node
    return (
        {"status": "unsupported_language", "language": detected_language},
        create_unsupported_language_node(),
    )


def create_initial_node(conversation_state: ConversationState) -> NodeConfig:
    """Create the initial conversation node that waits for user input."""
    supported_languages = "English, French, Spanish, or German"
    return {
        "name": "conversation",
        "role_messages": [
            {
                "role": "system",
                "content": f"You are a helpful assistant. Be casual and friendly. You support {supported_languages}. If the user speaks a language other than these, politely inform them that you only support {supported_languages} and end the conversation. Please don't use emojis. Wait for the user to speak first before responding. Write only one sentence per question asked.",
            },
        ],
        "task_messages": [
            {
                "role": "system",
                "content": f"Have a natural conversation only repond in max one or two sentences per question asked. start by telling hi",
            }
        ],
        # Remove task_messages to prevent bot from speaking first
        "functions": [handle_unsupported_language],
    }


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    """Run the multilingual conversation bot."""
    if not RIME_API_KEY:
        raise ValueError("RIME_API_KEY environment variable not set")
    if not DEEPGRAM_API_KEY:
        raise ValueError("DEEPGRAM_API_KEY environment variable not set")
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    # Create conversation state
    conversation_state = ConversationState()

    rtvi_processor = RTVIProcessor()
    rtvi_observer = RTVIObserver(rtvi_processor)
    session = aiohttp.ClientSession()
    audio_buffer = AudioBufferProcessor()

    stt = DeepgramSTTService(
        api_key=DEEPGRAM_API_KEY,
        audio_passthrough=True,
        live_options=LiveOptions(
            language="multi",
        ),
    )
    default_voice_config = RIME_LANGUAGE_MAP[Language.EN]
    tts = RimeHttpTTSService(
        api_key=RIME_API_KEY,
        voice_id=default_voice_config["speakerId"],
        aiohttp_session=session,
        model=default_voice_config["modelId"],
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
            LanguageDetectorProcessor(conversation_state=conversation_state),
            context_aggregator.user(),
            llm,
            tts,
            rtvi_processor,
            transport.output(),
            audio_buffer,
            context_aggregator.assistant(),
        ]
    )

    @audio_buffer.event_handler("on_audio_data")
    async def on_audio_data(processor, audio, sample_rate, num_channels):
        filename = await prepare_audio_filename(prefix="multilingual")
        logger.info("Saving audio to %s", filename)
        await save_audio_file(audio, filename, sample_rate, num_channels)

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
        await audio_buffer.start_recording()
        task.add_observer(rtvi_observer)
        # Initialize flow manager but don't trigger bot to speak
        await flow_manager.initialize(create_initial_node(conversation_state))
        # Remove any initial LLM run frame that would make bot speak first

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(_transport, _client):
        logger.info("Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)
    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, TRANSPORT_PARAMS)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
