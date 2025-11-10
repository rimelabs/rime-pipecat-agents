# Rime Multilingual Agent - Dynamic Language Switching

This project demonstrates dynamic language switching with Rime TTS in a Pipecat voice agent. The example shows how to detect the user's language in real-time and automatically switch voices accordingly. It's a simple example using Pipecat Flows and a custom FrameProcessor for language detection.

## Features

- **Automatic Language Detection**: Detects user language using Deepgram's multi-language STT
- **Dynamic Voice Switching**: Automatically switches Rime TTS voices based on detected language
- **Supported Languages**: 
  - English (Andromeda voice)
  - Spanish (Sirius voice)
  - French (Destin voice)
  - German (Klaus voice with Mist v2 model)
- **Custom FrameProcessor**: Demonstrates how to build custom frame processing logic in Pipecat

## Default Configuration

- The script is configured to start with SimpleWebRTC by default.
- To use Daily and Twilio, provide the appropriate parameters and set up the necessary configurations.

## Additional Resources

- To understand Pipecat's core architecture, read more [here](https://docs.pipecat.ai/getting-started/core-concepts).
- For a deeper dive into the fundamentals, check out [this guide](https://docs.pipecat.ai/guides/fundamentals).
- To explore all available transport options provided by Pipecat, look [here](https://docs.pipecat.ai/server/services/transport/daily).
- To understand how the language switching logic works, read about [custom FrameProcessors](https://docs.pipecat.ai/guides/fundamentals/custom-frame-processor#custom-frameprocessor).

## How It Works

The language switching is implemented using a custom `LanguageDetectorProcessor` that:

1. **Intercepts transcription frames** from Deepgram STT
2. **Extracts language information** from Deepgram's multi-language detection
3. **Pushes TTS update frames** when language changes are detected
4. **Updates the Rime TTS service** with the appropriate voice and model for the detected language

The language detection logic only works with **Deepgram API** because it relies on Deepgram's multi-language detection capability. You can modify this logic to fit your use case with other STT providers that support language detection.

## Important Notes

- **Voice Model Compatibility**: If you're using the Rime TTS service, use the **Mist voice model** (`mistv2`) , as Arcana is not yet fully supported for all languages in the Rime TTS service.
- The current implementation maps languages as follows:
  ```python
  Language.EN: Andromeda voice with Arcana model
  Language.ES: Sirius voice with Arcana model
  Language.FR: Destin voice with Arcana model
  Language.DE: Klaus voice with Mist v2 model
  ```

## Setup Instructions

### Prerequisites

1. **Install uv** (Python package manager):
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
   cd rime-pipecat-agents/rime-multilingual-agent
   ```

2. **Set up the environment file**:
   ```bash
   cp .env.example .env
   ```

   Add the following keys to the `.env` file:
   - `RIME_API_KEY`: Obtain from [Rime](https://app.rime.ai/tokens/)
   - `DEEPGRAM_API_KEY`: Obtain from [Deepgram](https://console.deepgram.com/project)
   - `OPENAI_API_KEY`: Obtain from [OpenAI](https://platform.openai.com/settings/organization/api-keys)

3. **Set up the environment with uv**:
   ```bash
   uv sync
   ```

   This will create a virtual environment and install all dependencies automatically.

### Running the Agent

**Start the multilingual agent**:
```bash
uv run main.py
```

The agent will start with SimpleWebRTC and automatically detect and switch languages as users speak in different languages.

## Production Deployment

The script provided is intended for local use only and should not be used in production. Once you decide to share it with the world, you can explore deployment patterns in Pipecat by visiting [this guide](https://docs.pipecat.ai/deployment/overview).

