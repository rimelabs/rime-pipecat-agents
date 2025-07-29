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

3. **Create a virtual environment:**
   ```bash
   python -m venv venv
   ```

4. **Activate the virtual environment:**
   ```bash
   source venv/bin/activate
   ```

5. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage Instructions

To start the script with different transport options:

- **WebRTC (default):**
  ```bash
  python3 main.py
  ```
  or
  ```bash
  python3 main.py --transport webrtc
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




