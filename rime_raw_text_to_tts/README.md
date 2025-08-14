# Pipecat Rime Internal Testing Script

## Project Overview

This project is an internal testing script for Rime, designed to convert text into audio frames using Rime's advanced Text-to-Speech (TTS) capabilities. It supports three major transport layers provided by Pipecat: Twilio, Daily, and SimpleWebRTC. The architecture is designed to allow seamless switching between these transport options without altering the internal structure of the application.

By default, the script is configured to start with SimpleWebRTC. However, for using Daily and Twilio, you need to provide the appropriate parameters and set up the necessary configurations.

To understand Pipecat's core architecture, you can read more [here](https://docs.pipecat.ai/getting-started/core-concepts). For a deeper dive into fundamentals, check out [this guide](https://docs.pipecat.ai/guides/fundamentals). To explore all available transport options provided by Pipecat, look [here](https://docs.pipecat.ai/server/services/transport/daily).

## Setup Instructions

1. **Navigate to the project directory:**
   ```bash
   cd rime_raw_text_to_tts
   ```

2. **Set up the `.env` file:** Add `RIME_API_KEY` to it. You can obtain the API key from [here](https://app.rime.ai/tokens/).

3. **Set up the environment with uv:**
   ```bash
   uv sync
   ```

   This will create a virtual environment and install all dependencies automatically.

   To activate the virtual environment manually (optional):
   ```bash
   source .venv/bin/activate
   ```

## Usage Instructions

To start the script with different transport options:

- **WebRTC (default):**
  ```bash
  uv run python main.py
  ```
  or
  ```bash
  uv run python main.py --transport webrtc
  ```

- **Twilio:**
  ```bash
  python3 main.py --transport twilio
  ```
  *Note: Specific parameters are required to make Twilio work.*

- **Daily:**
  ```bash
  python3 main.py --transport daily
  ```
  *Note: Specific parameters are required to make Daily work.*

To record the bot's audio and store it in the recordings folder:
```bash
python3 main.py --record
```

To provide text for the bot to speak, you have two options:

1. **Direct text input** using the `--text` parameter:
```bash
python3 main.py --text "Hello, this is the text I want the bot to speak"
```

2. **Text file input** using the `--textfile` parameter:
```bash
python3 main.py --textfile path/to/your/text.txt
```
*Note: For text files, the file must be a .txt file and contain the text you want to convert to speech. Use paths relative to the script directory, for example:*
```bash
# If your text file is in the same directory:
python3 main.py --textfile text.txt

# If your text file is in a subdirectory:
python3 main.py --textfile data/speech.txt
```

If neither `--text` nor `--textfile` is provided, the script will use a default welcome message.

You can combine multiple options:
```bash
# Using text file:
python3 main.py --transport daily --record --textfile my_speech.txt

# Using direct text:
python3 main.py --transport daily --record --text "Hello, welcome to the meeting!"
```


TODO
[] add the ability to type in more text and send to TTS
[] merge this with the single main funcion at root.

