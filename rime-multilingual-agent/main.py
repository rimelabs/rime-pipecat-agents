import asyncio
import os
import sys
from builtins import ValueError, bool, int, str, tuple, list, globals
from pipecat.services.openai.llm import OpenAILLMService
from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.rime.tts import RimeHttpTTSService
from pipecat.transports.base_transport import BaseTransport, TransportParams
import aiohttp
from pipecat.processors.frameworks.rtvi import RTVIProcessor, RTVIObserver
from pipecat_flows import (
    FlowArgs,
    FlowManager,
    FlowResult,
    FlowsFunctionSchema,
    NodeConfig,
)
from deepgram import LiveOptions


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
    "fra": {"speakerId": "serrin_joseph", "modelId": "arcana", "lang": "fra"},
    "ger": {"speakerId": "bergmann_katharina", "modelId": "arcana", "lang": "ger"},
    "hin": {"speakerId": "taru", "modelId": "arcana", "lang": "hin"},
}


transport_params = {
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(),
    ),
}


class LanguageResult(FlowResult):
    language: str
    status: str


# Function handlers
async def detect_language(args: FlowArgs) -> tuple[LanguageResult, NodeConfig]:
    """Detect the language that user is speaking."""
    language = args["language"]
    result = LanguageResult(language=language, status="success")
    logger.info(f"Detected language: {language}")
    if language in RIME_LANGUAGE_MAP:
        next_node = create_language_specific_node(language)
    else:
        next_node = (
            create_language_not_supported_node()
        )  # FIXED: was language_not_supported_node()

    return result, next_node


async def end_conversation(args: FlowArgs) -> tuple[None, NodeConfig]:
    """Handle conversation end."""
    return None, create_end_node()


# Create function schemas
language_detection_schema = FlowsFunctionSchema(
    name="detect_language",
    description="Detect the language that user is speaking.",
    properties={
        "language": {"type": "string", "enum": ["eng", "spa", "fra", "ger", "hin"]}
    },
    required=["language"],
    handler=detect_language,
)

end_conversation_schema = FlowsFunctionSchema(
    name="end_conversation",
    description="End the conversation",
    properties={},
    required=[],
    handler=end_conversation,
)


# Node configurations
def create_initial_node() -> NodeConfig:
    """Create initial node for detecting the language that user is speaking."""
    return {
        "name": "initial",
        "role_messages": [
            {
                "role": "system",
                "content": "You are a language detector and a restaurant reservation assistant for La Maison. Be casual and friendly. This is a voice conversation, so avoid special characters and emojis.",
            }
        ],
        "task_messages": [
            {
                "role": "system",
                "content": "Your task is to detect the language the user is speaking in.",
            }
        ],
        "functions": [language_detection_schema],
        "respond_immediately": False,
    }


async def update_tts_language(action: dict, flow_manager: FlowManager):
    """Update TTS settings based on detected language."""
    from pipecat.frames.frames import TTSUpdateSettingsFrame

    language = action.get("language")
    lang_config = RIME_LANGUAGE_MAP.get(language, RIME_LANGUAGE_MAP["eng"])

    await flow_manager.task.queue_frame(
        TTSUpdateSettingsFrame(
            settings={
                "voice_id": lang_config["speakerId"],
                "model": lang_config["modelId"],
            }
        )
    )

    flow_manager.state["current_language"] = language
    logger.info(
        f"Updated TTS to language: {language} with voice {lang_config['speakerId']}"
    )


async def reset_tts_to_default(action: dict, flow_manager: FlowManager):
    """Reset TTS settings to default English configuration."""
    from pipecat.frames.frames import TTSUpdateSettingsFrame

    default_config = RIME_LANGUAGE_MAP["eng"]

    await flow_manager.task.queue_frame(
        TTSUpdateSettingsFrame(
            settings={
                "voice_id": default_config["speakerId"],
                "model": default_config["modelId"],
            }
        )
    )

    flow_manager.state["current_language"] = "eng"
    logger.info(f"Reset TTS to default English voice: {default_config['speakerId']}")


def create_language_specific_node(language: str) -> NodeConfig:
    """Create a node with language-specific configuration."""
    lang_config = RIME_LANGUAGE_MAP.get(language, RIME_LANGUAGE_MAP["eng"])
    logger.debug(f"Creating language-specific node for {language}: {lang_config}")
    return {
        "name": "restaurant_manager",
        "task_messages": [
            {
                "role": "system",
                "content": f"You are a restaurant reservation assistant for La Maison. Be casual and friendly. This is a voice conversation, so avoid special characters and emojis. Talk to them in the {language} language they are speaking in. Help them with their reservation and when done, end the conversation.",
            }
        ],
        "functions": [language_detection_schema,end_conversation_schema],  # ADDED: function to end conversation
        "post_actions": [
            {"type": "function", "handler": update_tts_language, "language": language}
        ],
    }


def create_language_not_supported_node() -> NodeConfig:
    """Create a node for when the language is not supported."""
    return {
        "name": "language_not_supported",
        "task_messages": [
            {
                "role": "system",
                "content": "The language you are speaking in is not supported. Please speak in English, Spanish, French, German, or Hindi. After informing them, end the conversation.",
            }
        ],
        "functions": [language_detection_schema,end_conversation_schema],  # ADDED: function to end conversation
        "pre_actions": [{"type": "function", "handler": reset_tts_to_default}],
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


# Main setup
async def run_bot(
    transport: BaseTransport, runner_args: RunnerArguments, wait_for_user: bool = False
):
    """Run the restaurant reservation bot."""
    if not RIME_API_KEY:
        raise ValueError("RIME_API_KEY environment variable not set")
    if not DEEPGRAM_API_KEY:
        raise ValueError("DEEPGRAM_API_KEY environment variable not set")
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable not set")

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
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        print("Client connected")
        task.add_observer(rtvi_observer)
        await flow_manager.initialize(create_initial_node())

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)
    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    wait_for_user = globals().get("WAIT_FOR_USER", False)
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args, wait_for_user)


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Restaurant reservation bot")
    parser.add_argument(
        "--wait-for-user",
        action="store_true",
        help="If set, the bot will wait for the user to speak first",
    )

    args, remaining = parser.parse_known_args()
    WAIT_FOR_USER = args.wait_for_user

    if "--wait-for-user" in sys.argv:
        sys.argv.remove("--wait-for-user")

    from pipecat.runner.run import main

    main()
