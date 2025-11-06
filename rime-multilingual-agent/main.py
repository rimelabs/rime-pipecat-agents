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
from pipecat.frames.frames import TTSUpdateSettingsFrame
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.rime.tts import RimeHttpTTSService
from pipecat.transcriptions.language import Language
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
from openai import AsyncOpenAI
import json
from pipecat.frames.frames import (
    Frame,
    TranscriptionFrame,
    SystemFrame,
    UserStoppedSpeakingFrame,
    EndFrame,
)
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from openai import AsyncOpenAI
import json


load_dotenv(override=True)
RIME_VOICE_ID = "astra"
RIME_MODEL = "arcana"
RIME_URL = "wss://users-ws.rime.ai/ws2"

RIME_API_KEY = os.getenv("RIME_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
language_detected = "eng"
RIME_LANGUAGE_MAP = {
    "eng": {"speakerId": "andromeda", "modelId": "arcana", "lang": Language.EN},
    "spa": {"speakerId": "sirius", "modelId": "arcana", "lang": Language.ES},
    "fra": {"speakerId": "serrin_joseph", "modelId": "arcana", "lang": Language.FR},
    "ger": {
        "speakerId": "bergmann_katharina",
        "modelId": "arcana",
        "lang": Language.DE,
    },
}


transport_params = {
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(),
    ),
}


# Node configurations
def create_initial_node() -> NodeConfig:
    global language_detected
    return {
        "name": "conversation",
        "role_messages": [
            {
                "role": "system",
                "content": f"You are a helpful assistant. Be casual and friendly. {language_detected} if it other than english, french , spanish , german then say you are not able to understand them and end the conversation.",
            }
        ],
        "task_messages": [
            {"role": "system", "content": f"Have a natural conversation with the user in the language they are speaking in. {language_detected} if it other than english, french , spanish , german then say you are not able to understand them and end the conversation."}
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
    def __init__(self, api_key: str):
        super().__init__()
        self._client = AsyncOpenAI(api_key=api_key)
        self._frame_buffer = []
        self._language_detected = False

    async def _detect_language(self, text: str) -> dict:
        """Make OpenAI API call to detect language."""
        response = await self._client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Detect the language of the user's text. Respond with JSON 'lang' for eng, spa, fra, ger",
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
        """Process frames and detect language before passing to LLM."""
        await super().process_frame(frame, direction)

        # Handle UserStoppedSpeakingFrame and EndFrame FIRST
        if isinstance(frame, (UserStoppedSpeakingFrame, EndFrame)):
            # User finished speaking OR pipeline ending - now detect language
            if self._frame_buffer:
                # Combine all buffered text
                full_text = " ".join(
                    f.text
                    for f, _ in self._frame_buffer
                    if isinstance(f, TranscriptionFrame)
                )
                logger.info(f"Full text: {full_text}")
                language_result = await self._detect_language(full_text)
                print(f"Detected language: {language_result}")
                detected_lang = language_result.get("lang")
                if detected_lang in RIME_LANGUAGE_MAP:
                    global language_detected
                    lang_config = RIME_LANGUAGE_MAP[detected_lang]
                    tts_update_frame = TTSUpdateSettingsFrame(
                        settings={
                            "voice_id": lang_config["speakerId"],
                            "model": lang_config["modelId"],
                            "language": lang_config["lang"],
                        }
                    )
                    language_detected = detected_lang
                    await self.push_frame(tts_update_frame, direction)
                    logger.info(f"Updated TTS settings for language: {detected_lang}")

                # Now push all buffered frames downstream to LLM
                for buffered_frame, buffered_direction in self._frame_buffer:
                    await self.push_frame(buffered_frame, buffered_direction)
                self._frame_buffer.clear()

            # Push the UserStoppedSpeakingFrame or EndFrame
            await self.push_frame(frame, direction)

        # Then handle other system frames
        elif isinstance(frame, SystemFrame):
            await self.push_frame(frame, direction)

        elif isinstance(frame, TranscriptionFrame):
            # Keep buffering ALL transcription frames
            self._frame_buffer.append((frame, direction))
            # Don't push yet - wait for user to stop speaking

        else:
            # Pass through all other frames
            await self.push_frame(frame, direction)


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
            LanguageDetectionProcessor(api_key=OPENAI_API_KEY),
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
