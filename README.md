# Rime-Pipecat Agents Demo Repository

A collection of demo agents showcasing the integration of [Rime](https://rime.ai) and [Pipecat](https://pipecat.ai) for building powerful voice AI applications. This repository demonstrates various patterns for creating conversational agents with speech-to-text (STT), large language models (LLM), and text-to-speech (TTS) capabilities.

## 📚 Examples

### 1. [rime-simple-agent](./rime-simple-agent/)

**Simple STT → LLM → TTS Integration**

A straightforward implementation demonstrating the basic voice chat pipeline. This example is perfect for understanding the fundamentals of integrating Rime's TTS with Pipecat's framework.

**Features:**
- Simple voice chat agent with back-and-forth communication
- Support for multiple transport layers (Twilio, Daily, SimpleWebRTC)
- Console mode for local testing
- Recording capabilities
- Basic SST → LLM → TTS pipeline

**Best for:** Getting started with Rime and Pipecat integration, understanding the basic architecture.

---

### 2. [rime-multilingual-agent](./rime-multilingual-agent/)

**Advanced Multilingual Agent with Pipecat Flows**

A more sophisticated example demonstrating the power of Pipecat Flows for building complex conversational agents. This example showcases real-time language switching based on user speech.

**Features:**
- Advanced STT → LLM → TTS pipeline with Pipecat Flows
- Real-time language detection and switching
- Automatic adaptation to user's spoken language
- Demonstrates Pipecat Flow patterns for complex conversation management
- Production-ready conversation state management

**Best for:** Building production-grade multilingual conversational agents, understanding Pipecat Flows, implementing language-aware applications.

---

## 🚀 Quick Start

Each example has its own detailed README with setup instructions. Choose the example that best fits your needs:

- **New to Rime and Pipecat?** Start with [rime-simple-agent](./rime-simple-agent/)
- **Building a multilingual application?** Check out [rime-multilingual-agent](./rime-multilingual-agent/)

## 🔧 Common Prerequisites

All examples require:

1. **Python 3.9+** with [uv](https://github.com/astral-sh/uv) package manager
2. **API Keys:**
   - [Rime API Key](https://app.rime.ai/tokens/)
   - [Deepgram API Key](https://console.deepgram.com/project) (for STT)
   - [OpenAI API Key](https://platform.openai.com/settings/organization/api-keys) (for LLM)

## 📖 Resources

- **Rime Documentation:** [https://docs.rime.ai](https://docs.rime.ai)
- **Pipecat Documentation:** [https://docs.pipecat.ai](https://docs.pipecat.ai)
- **Pipecat Core Concepts:** [Getting Started Guide](https://docs.pipecat.ai/getting-started/core-concepts)
- **Pipecat Flows:** [Flows Documentation](https://docs.pipecat.ai/guides/flows)

## 🏗️ Architecture

All examples follow a similar high-level architecture:

```
User Voice Input → STT (Deepgram) → LLM (OpenAI) → TTS (Rime) → Voice Output
```

The key differences lie in:
- **Simple Agent:** Basic pipeline implementation
- **Multilingual Agent:** Enhanced with Pipecat Flows for language detection and complex conversation management

## ⚠️ Important Notes

- These examples are intended for **development and demonstration purposes only**
- For production deployments, refer to [Pipecat's deployment guide](https://docs.pipecat.ai/deployment/overview)
- Each example can run locally or be deployed to various platforms

## 🤝 Contributing

This is a demo repository showcasing integration patterns. Feel free to use these examples as starting points for your own projects.

## 📄 License

See individual example directories for licensing information.

---

**Ready to build voice AI agents?** Pick an example above and start building! 🎙️