# Rime Pipecat Agent (SST -> LLM -> TTS)

This project is a Rime Pipecat hosted agent demonstrating the SST -> LLM -> TTS implementation using Pipecat. It enables voice chat with the agent, allowing for back-and-forth communication. The project supports three major transport layers provided by Pipecat: Twilio, Daily, and SimpleWebRTC. The architecture is designed to allow seamless switching between these transport options without altering the internal structure of the application.

## Default Configuration

- The script is configured to start with SimpleWebRTC by default.
- To use Daily and Twilio, provide the appropriate parameters and set up the necessary configurations.

## Additional Resources

- To understand Pipecat's core architecture, read more [here](https://docs.pipecat.ai/getting-started/core-concepts).
- For a deeper dive into the fundamentals, check out [this guide](https://docs.pipecat.ai/guides/fundamentals).
- To explore all available transport options provided by Pipecat, look [here](https://docs.pipecat.ai/server/services/transport/daily).

## Architecture Diagram

![Architecture Diagram](path/to/placeholder-image.png)

*Note: Replace the placeholder image path with the actual image path once available.*

