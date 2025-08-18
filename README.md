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

<img width="1393" height="142" alt="image" src="https://github.com/user-attachments/assets/683b6ff7-41e5-411c-80c4-511171f73ea8" />


## Setup Instructions

### Prerequisites

**Install uv** (Python package manager):
```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with pip
pip install uv

# Or with Homebrew (macOS)
brew install uv
```

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/rimelabs/Rime-pipecate-agents.git
   # or
   git clone git@github.com:rimelabs/Rime-pipecate-agents.git
   cd rime-pipecate-agents
   ```

2. **Set up the environment file**:
   ```bash
   cp .env.example .env
   ```

   Add the following keys to the `.env` file:
   - `RIME_API_KEY`: Obtain from [Rime](https://app.rime.ai/tokens/)
   - `DEEPGRAM_API_KEY`: Obtain from [Deepgram](https://console.deepgram.com/project)
   - `OPENAI_API_KEY`: Obtain from [OpenAI](https://platform.openai.com/settings/organization/api-keys)

4. **Set up the environment with uv**:
   ```bash
   uv sync
   ```

   This will create a virtual environment and install all dependencies automatically.

**Run the Rime Agent**:

1. **Web App Demo**:
   ```bash
   uv run rime_agent.py
   ```

2. **Console Mode**:
   The console mode allows you to hear text-to-speech directly from your terminal without needing a web interface.

   ```bash
   # Convert text to speech directly
   uv run rime_agent.py --text "your text here"

   # Convert contents of a text file to speech
   uv run rime_agent.py --text-file path/to/your/file.txt

   # Run with default sample text
   uv run rime_agent.py --console

   # Record the audio output (works with any of the above)
   uv run rime_agent.py --text "your text" --record
   ```

   When using --record, the audio will be saved in the /recordings directory with a timestamp.

3. **Recording Mode**:
   ```bash
   uv run rime_agent.py --record
   ```
   Then upon clicking "disconnect" in the UI the conversation will be saved to an audio file in the /recordings directory.

## Important Note

The script provided is intended for local use only and should not be used in production. Once you decide to share it with the world, you can explore deployment patterns in Pipecat by visiting [this guide](https://docs.pipecat.ai/guides/deployment/overview).
