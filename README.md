# WeatherDetection LangChain Agent

A LangChain AI agent built by **Sandeep** that uses the ReAct reasoning loop to answer weather queries and perform math calculations using tools.

## What it does
- Detects weather for cities using a `get_weather` tool
- Evaluates math expressions using a `calculate_expression` tool
- Works with **Gemini, GPT-4o, or Claude** (via API key) or runs fully locally with a simulated model

## How to run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. (Optional) Set an API key for a real LLM
```bash
# Google Gemini
set GOOGLE_API_KEY=your-key-here

# OR OpenAI
set OPENAI_API_KEY=your-key-here

# OR Anthropic Claude
set ANTHROPIC_API_KEY=your-key-here
```

### 3. Run the agent
```bash
python run_agent_demo.py
```

> No API key? No problem — the agent runs with a built-in simulated model locally.

## Project files
| File | Description |
|---|---|
| `run_agent_demo.py` | Main agent code |
| `data/nifty50_24nov2023_to_1feb2024.png` | NIFTY 50 market data (Nov 2023 – Feb 2024) |
| `requirements.txt` | Python dependencies |

## Built with
- [LangChain](https://github.com/langchain-ai/langchain)
- [LangGraph](https://github.com/langchain-ai/langgraph)
