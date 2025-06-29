# AI Debate Simulator using LangGraph + Groq

This project simulates a structured debate between two AI agents using the [LangGraph](https://python.langchain.com/docs/langgraph/) framework, LLm is loaded using ollama.

Each agent takes on a role (e.g., Scientist vs. Philosopher) and debates a given topic through several turns. A third AI (the Judge) evaluates the conversation and declares a winner.

---

## Features

-  **Multi-turn Debate**: Two agents alternate turns for up to 3 rounds (6 turns).
-  **Dynamic Memory**: Each agent only sees the opponent's most recent response.
-  **AI Judge**: An impartial model summarizes and declares the winner.
-  **Structured LangGraph Flow**: Fully modular LangGraph workflow with state transitions.
-  **Logging & Debugging**: All state transitions and model outputs logged to `state_log.log`.

---

## Instruction


