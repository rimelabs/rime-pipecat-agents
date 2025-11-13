# Rime-Pipecat Agents Demo Repository

A collection of demo agents showcasing the integration of [Rime](https://rime.ai) and [Pipecat](https://pipecat.ai) for building powerful voice AI applications. This repository demonstrates various patterns for creating conversational agents with speech-to-text (`STT`), large language models (`LLM`), and text-to-speech (`TTS`) capabilities.

## 📚 Examples

### 1. [rime-simple-agent](./rime-simple-agent/)

**Simple `STT` → `LLM` → `TTS` Integration**

A straightforward implementation demonstrating the basic voice chat pipeline. This example is perfect for understanding the fundamentals of integrating Rime's `TTS` with Pipecat's framework.

**Features:**
- Simple voice chat agent with back-and-forth communication
- Support for multiple transport layers (`Twilio`, `Daily`, `SimpleWebRTC`)
- Console mode for local testing
- Recording capabilities
- Basic `STT` → `LLM` → `TTS` pipeline

**Best for:** Getting started with Rime and Pipecat integration, understanding the basic architecture.

---

### 2. [rime-multilingual-agent](./rime-multilingual-agent/)

**Dynamic Language Switching with Custom `FrameProcessor`**

An advanced example demonstrating real-time language detection and automatic voice switching. This agent uses a custom `FrameProcessor` to detect the user's language and dynamically switches Rime TTS voices accordingly.

**Features:**
- Automatic language detection using Deepgram's multi-language `STT`
- Dynamic voice switching based on detected language
- Support for 4 languages: English, Spanish, French, and German
- Custom `LanguageDetectorProcessor` implementation
- Uses Pipecat `Flows` for conversation management
- Demonstrates advanced frame processing patterns

**Best for:** Learning how to build custom `FrameProcessor`s, implementing multilingual voice agents, understanding dynamic `TTS` configuration.

---
