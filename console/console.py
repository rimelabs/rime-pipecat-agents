#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import os
import sys

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.frames.frames import EndFrame, TTSSpeakFrame
from pipecat.services.rime.tts import RimeTTSService
from pipecat.transcriptions.language import Language
from pipecat.transports.local.audio import (
    LocalAudioTransport,
    LocalAudioTransportParams,
)

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


async def main():
    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
        )
    )

    api_key = os.getenv("RIME_API_KEY")
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

    default_text = "There's a 2022 Ferrari F8 Tributo with 7,638 miles, a 2018 Ferrari 488 G. T. B. with 9,837 miles, and a 2019 Ferrari G. T. C. 4 Lusso V12 with 17,097 miles."

    pipeline = Pipeline([transport.input(), tts, transport.output()])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    await task.queue_frames(
        [
            TTSSpeakFrame(default_text),
            EndFrame(),
        ]
    )

    runner = PipelineRunner()

    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
