import asyncio
import os
from dotenv import load_dotenv

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.rime.tts import RimeHttpTTSService
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.transports.local.audio import (
    LocalAudioTransport,
    LocalAudioTransportParams,
)
import aiohttp

load_dotenv()


async def main():
    # Initialize services
    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")

    session = aiohttp.ClientSession()
    tts = RimeHttpTTSService(
        api_key=os.getenv("RIME_API_KEY"),
        voice_id="glacier",
        model="mistv2",
        aiohttp_session=session,
    )

    # Create context
    context = OpenAILLMContext(  
    messages=[{  
        "role": "system",  
            "content": "You are a helpful assistant. Keep responses brief.",
        }
    ]
    )
    context_aggregator = llm.create_context_aggregator(context)  

    # Create transport
    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
        )
    )

    # Build pipeline
    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            context_aggregator.user(),
            llm,
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(pipeline, params=PipelineParams(enable_metrics=True))

    runner = PipelineRunner()

    try:
        await runner.run(task)
    finally:
        await session.close()


if __name__ == "__main__":
    asyncio.run(main())
