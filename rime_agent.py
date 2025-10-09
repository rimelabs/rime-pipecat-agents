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
from pipecat.transports.base_transport import TransportParams  
import aiohttp  
  
load_dotenv()  
  
  
async def bot(runner_args):  
    """Bot entry point for the Pipecat runner."""  
    # Initialize services  
    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))  
      
    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")  
      
    session = aiohttp.ClientSession()  
    tts = RimeHttpTTSService(  
        api_key=os.getenv("RIME_API_KEY"),  
        voice_id="eva",  
        model="mistv2",  
        sample_rate=8000,
        aiohttp_session=session,  
    )  
      
    # Create context  
    context = OpenAILLMContext(  
        messages=[{  
            "role": "system",  
            "content": "You are a helpful assistant. Keep responses brief."  
        }]  
    )  
    context_aggregator = llm.create_context_aggregator(context)  
      
    # Create transport from runner args  
    from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport  
    transport = SmallWebRTCTransport(  
        runner_args.webrtc_connection,  
        TransportParams(  
            audio_in_enabled=True,  
            audio_out_enabled=True,  
            vad_analyzer=SileroVADAnalyzer()  
        )  
    )  
      
    # Build pipeline  
    pipeline = Pipeline([  
        transport.input(),  
        stt,  
        context_aggregator.user(),  
        llm,  
        tts,  
        transport.output(),  
        context_aggregator.assistant(),  
    ])  
      
    task = PipelineTask(pipeline, params=PipelineParams(enable_metrics=True))  
      
    runner = PipelineRunner()  
      
    try:  
        await runner.run(task)  
    finally:  
        await session.close()  
  
  
if __name__ == "__main__":  
    from pipecat.runner.run import main  
    main()