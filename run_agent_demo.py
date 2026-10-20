"""LangChain Agent — Built by Sandeep.

A hands-on LangChain agent I built to learn and demonstrate how AI agents work:
1. Define custom tools with type annotations and docstrings.
2. Initialize the chat model (live LLM or local simulated model).
3. Create the agent with `create_agent` and wire it up with tools.
4. Stream the step-by-step reasoning and tool execution loop in real time.

Author : Sandeep
Purpose: Personal learning project — exploring LangChain agents and the ReAct loop.
"""

from __future__ import annotations

import os
from typing import Any, Sequence
from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool


# 1. Define custom tools
@tool
def get_weather(city: str) -> str:
    """Get the current weather forecast for a given city."""
    forecasts = {
        "tokyo": "Clear skies, 18°C (64°F)",
        "new york": "Sunny and crisp, 15°C (59°F)",
        "london": "Overcast with light showers, 12°C (54°F)",
        "paris": "Mild and pleasant, 16°C (61°F)",
        "mumbai": "Warm and humid, 30°C (86°F)",
    }
    return forecasts.get(city.lower().strip(), f"70°F and sunny in {city}")


@tool
def calculate_expression(expression: str) -> str:
    """Evaluate a mathematical calculation expression safely."""
    allowed = set("0123456789+-*/(). ")
    if not all(c in allowed for c in expression):
        return f"Error: Expression contains unsupported characters: {expression}"
    try:
        return str(eval(expression, {"__builtins__": None}, {}))  # noqa: S307
    except Exception as exc:
        return f"Calculation failed: {exc}"


# 2. Simulated Model supporting Tool-Calling for local execution
class SimulatedAgentModel(BaseChatModel):
    """Local simulated model that demonstrates the ReAct tool-calling loop."""

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        last_msg = messages[-1]
        if isinstance(last_msg, ToolMessage):
            final_content = (
                f"I received the result from the tool ({last_msg.name}):\n"
                f"-> \"{last_msg.content}\"\n\n"
                f"Final Answer: The requested operation was successfully completed."
            )
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=final_content))])

        user_prompt = last_msg.content if isinstance(last_msg.content, str) else str(last_msg.content)
        lower_prompt = user_prompt.lower()

        if "weather" in lower_prompt:
            city = "Tokyo"
            for candidate in ["tokyo", "new york", "london", "paris", "mumbai"]:
                if candidate in lower_prompt:
                    city = candidate.title()
                    break
            tool_call = ToolCall(
                name="get_weather",
                args={"city": city},
                id="call_weather_001",
            )
            ai_msg = AIMessage(
                content=f"I need to check the weather in {city}. Calling the weather tool...",
                tool_calls=[tool_call],
            )
            return ChatResult(generations=[ChatGeneration(message=ai_msg)])

        if "calculate" in lower_prompt or any(op in lower_prompt for op in ["*", "+", "/", "-"]):
            expr = "42 * 7 + 10"
            tool_call = ToolCall(
                name="calculate_expression",
                args={"expression": expr},
                id="call_calc_001",
            )
            ai_msg = AIMessage(
                content=f"I will calculate the arithmetic expression '{expr}'. Calling calculator...",
                tool_calls=[tool_call],
            )
            return ChatResult(generations=[ChatGeneration(message=ai_msg)])

        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        content=f"Hello! I am an agent powered by LangChain. You said: '{user_prompt}'."
                    )
                )
            ]
        )

    @property
    def _llm_type(self) -> str:
        return "simulated-agent-model"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> Any:
        return self.bind(**kwargs)


def main() -> None:
    print("=" * 60)
    print("   LANGCHAIN AGENT — BUILT BY SANDEEP   ")
    print("=" * 60)

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    if api_key:
        print("[*] Found API key in environment. Initializing live model...")
        provider = "openai:gpt-4o" if os.getenv("OPENAI_API_KEY") else "anthropic:claude-3-5-sonnet-20241022"
        from langchain.chat_models import init_chat_model
        model = init_chat_model(provider)
    else:
        print("[*] No external API key found. Using simulated tool-calling model.")
        print("    (This demonstrates the full LangChain ReAct agent loop locally)")
        model = SimulatedAgentModel()

    print("\n[1] Creating agent with tools: [get_weather, calculate_expression]...")
    agent = create_agent(
        model=model,
        tools=[get_weather, calculate_expression],
        system_prompt="You are a helpful AI assistant capable of looking up weather and computing math.",
    )
    print("    Agent compiled successfully as a StateGraph!")

    query1 = "What is the weather in Tokyo today?"
    print(f"\n[2] Invoking Agent with Query: \"{query1}\"")
    print("-" * 60)

    for event in agent.stream(
        {"messages": [HumanMessage(content=query1)]},
        stream_mode="updates",
    ):
        for node_name, updates in event.items():
            print(f"[{node_name.upper()} STEP]:")
            messages = updates.get("messages", [])
            for msg in messages:
                if isinstance(msg, AIMessage):
                    if msg.tool_calls:
                        print(f"  AI Decision: {msg.content}")
                        for tc in msg.tool_calls:
                            print(f"  Tool Call -> {tc['name']} with arguments: {tc['args']}")
                    else:
                        print(f"  AI Final Response:\n{msg.content}")
                elif isinstance(msg, ToolMessage):
                    print(f"  Tool Output ({msg.name}): {msg.content}")

    query2 = "Can you calculate 42 * 7 + 10?"
    print(f"\n[3] Invoking Agent with Query: \"{query2}\"")
    print("-" * 60)

    for event in agent.stream(
        {"messages": [HumanMessage(content=query2)]},
        stream_mode="updates",
    ):
        for node_name, updates in event.items():
            print(f"[{node_name.upper()} STEP]:")
            messages = updates.get("messages", [])
            for msg in messages:
                if isinstance(msg, AIMessage):
                    if msg.tool_calls:
                        print(f"  AI Decision: {msg.content}")
                        for tc in msg.tool_calls:
                            print(f"  Tool Call -> {tc['name']} with arguments: {tc['args']}")
                    else:
                        print(f"  AI Final Response:\n{msg.content}")
                elif isinstance(msg, ToolMessage):
                    print(f"  Tool Output ({msg.name}): {msg.content}")

    print("\n" + "=" * 60)
    print("  AGENT RUN COMPLETE  |  Project by Sandeep")
    print("=" * 60)


if __name__ == "__main__":
    main()
