import asyncio  
import os  
import atexit
import signal
from dotenv import load_dotenv  
from loguru import logger

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

# Import our logging configuration
from logging_config import setup_application_logging, log_session_end

load_dotenv()

# Set up logging at module level
current_log_file = setup_application_logging()
logger.info(f"Rime Agent starting up. Logs will be written to: {current_log_file}")

# Register cleanup function for graceful shutdown
def cleanup_on_exit():
    """Clean up resources and log session end."""
    logger.info("Application shutting down...")
    log_session_end()

atexit.register(cleanup_on_exit)

# Handle SIGINT (Ctrl+C) gracefully
def signal_handler(signum, _frame):
    """Handle interrupt signals gracefully."""
    logger.info(f"Received signal {signum}. Initiating graceful shutdown...")
    cleanup_on_exit()
    exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
  
  
async def bot(runner_args):  
    """Bot entry point for the Pipecat runner."""  
    logger.info("Initializing bot services...")
    
    # Initialize services  
    logger.debug("Setting up Deepgram STT service...")
    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))  
      
    logger.debug("Setting up OpenAI LLM service...")
    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")  
      
    logger.debug("Creating aiohttp session...")
    session = aiohttp.ClientSession()  
    
    logger.debug("Setting up Rime TTS service...")
    tts = RimeHttpTTSService(  
        api_key=os.getenv("RIME_API_KEY"),  
        voice_id="eva",  
        model="mistv2",  
        sample_rate=8000,
        aiohttp_session=session,  
    )  
      
    # Create context  
    logger.debug("Setting up OpenAI LLM context...")
    context = OpenAILLMContext(  
        messages=[{  
            "role": "system",  
            "content": "You are a helpful assistant. Keep responses brief."  
        }]  
    )  
    context_aggregator = llm.create_context_aggregator(context)  
      
    # Create transport from runner args  
    logger.debug("Setting up WebRTC transport...")
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
    logger.info("Building processing pipeline...")
    pipeline = Pipeline([  
        transport.input(),  
        stt,  
        context_aggregator.user(),  
        llm,  
        tts,  
        transport.output(),  
        context_aggregator.assistant(),  
    ])  
      
    logger.debug("Creating pipeline task with metrics enabled...")
    task = PipelineTask(pipeline, params=PipelineParams(enable_metrics=True))  
      
    logger.debug("Creating pipeline runner...")
    runner = PipelineRunner()  
      
    try:  
        logger.info("Starting pipeline runner - Bot is now active!")
        await runner.run(task)  
    except Exception as e:
        logger.error(f"Error running pipeline: {e}")
        raise
    finally:  
        logger.info("Closing aiohttp session...")
        await session.close()
        logger.info("Bot session ended.")  
  
  
if __name__ == "__main__":  
    from pipecat.runner.run import main  
    main()