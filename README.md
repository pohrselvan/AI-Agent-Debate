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
1. Clone github repo
   ```bash
   git clone https://github.com/pohrselvan/AI-Agent-Debate

2. Run the ollama
   ```bash
   ollama run llama-3.2

3. Run the `AI_debate_using_ollama.py`
   ```bash
   python3 AI_debate_using_ollama.py

## WorkFlow 
![Task-3](https://github.com/user-attachments/assets/5ed246ad-0226-4853-ae5c-e56ac6639a85)
