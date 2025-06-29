from typing import TypedDict, List, Literal, Tuple, Annotated
from langgraph.graph import StateGraph, START, END 
import logging 
from langchain_core.prompts import ChatPromptTemplate 
from langchain_core.output_parsers import StrOutputParser 
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
import os
import operator
from langchain_community.llms import Ollama


# Setup logging
logging.basicConfig(
    filename="state_log.log",
    filemode="a", 
    level=logging.INFO, 
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# Define AgentState
class AgentState(TypedDict):
    topic: str
    agent_a: str
    agent_b: str
    round: int
    turn: int
    result: str
    memory: Annotated[List[Tuple[str, str]], operator.add]

# Initialize parser
parser = StrOutputParser()

# Node: Initiate
def initiate_node(state: AgentState) -> AgentState:
    state["round"] = 1
    state["turn"] = 1  # Track total turns
    state["memory"] = []
    logging.info(f"[Initiate Node] Debate topic: {state['topic']}")
    logging.info(f"[Initiate Node] Role of Agent A: {state['agent_a']}")
    logging.info(f"[Initiate Node] Role of Agent B: {state['agent_b']}")
    return state

# Node: Round Controller (decides which agent goes)
def round_controller(state: AgentState) -> str:
    # Agent A goes on odd turns, Agent B on even turns
    if state["turn"] % 2 == 1:
        return "Call_Agent_A"
    else:
        return "Call_Agent_B"

# Node: Agent A
def Agent_A(state: AgentState) -> AgentState:
    system_prompt = f"""You are a {state['agent_a']} in a debate on: '{state['topic']}'. Give precise, focused responses in maximum 50 words. Be direct and clear."""
    
    llm1 = Ollama(model = "llama3.2") # Switched to smaller, faster model
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "{user_input}")
    ])
    agent_a = prompt | llm1 | parser

    if state['turn'] == 1:
        # First turn - opening statement
        user_prompt = f"Topic: {state['topic']}\nGive your opening argument in max 50 words."
    else:
        # Build context from limited memory (only last 2 exchanges)
        context = ""
        memory = state.get("memory", [])
        if memory:
            # Get only the most recent opponent response
            latest_opponent = None
            for role, content in reversed(memory):
                if role == "agent_b":
                    latest_opponent = content[:200] + "..." if len(content) > 200 else content  # Truncate long responses
                    break
            
            if latest_opponent:
                context = f"Opponent said: {latest_opponent}\n"
        
        user_prompt = f"{context}Your counter-argument (max 50 words):"

    try:
        state["result"] = agent_a.invoke({"user_input": user_prompt})
        logging.info(f"[Agent A Turn {state['turn']}] {state['result']}")
        print(f"Agent A (Turn {state['turn']}): {state['result']}")
    except Exception as e:
        logging.error(f"[Agent A Error] {e}")
        state["result"] = f"Agent A cannot respond due to API limits. Proceeding with debate."
        print(f"Agent A (Turn {state['turn']}): {state['result']}")
    
    return state

# Node: Agent B
def Agent_B(state: AgentState) -> AgentState:
    system_prompt = f"""You are a {state['agent_b']} in a debate on: '{state['topic']}'. Give precise, focused responses in maximum 50 words. Be direct and clear."""
    
    llm2 = Ollama(model = "llama3.2")  # Switched to smaller, faster model
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "{user_input}")
    ])
    agent_b = prompt | llm2 | parser

    # Build context from limited memory (only last 2 exchanges)
    context = ""
    memory = state.get("memory", [])
    if memory:
        # Get only the most recent opponent response
        latest_opponent = None
        for role, content in reversed(memory):
            if role == "agent_a":
                latest_opponent = content[:200] + "..." if len(content) > 200 else content  # Truncate long responses
                break
        
        if latest_opponent:
            context = f"Opponent said: {latest_opponent}\n"
    
    user_prompt = f"{context}Your counter-argument (max 50 words):"

    try:
        state["result"] = agent_b.invoke({"user_input": user_prompt})
        logging.info(f"[Agent B Turn {state['turn']}] {state['result']}")
        print(f"Agent B (Turn {state['turn']}): {state['result']}")
    except Exception as e:
        logging.error(f"[Agent B Error] {e}")
        state["result"] = f"Agent B cannot respond due to API limits. Proceeding with debate."
        print(f"Agent B (Turn {state['turn']}): {state['result']}")
    
    return state

# Node: Memory Updater
def Memory_Node(state: AgentState) -> AgentState:
    # Create new memory list to avoid accumulation issues
    current_memory = list(state.get("memory", []))
    
    # Determine which agent just spoke based on turn number
    if state["turn"] % 2 == 1:
        # Odd turn = Agent A spoke
        current_memory.append(("agent_a", state["result"]))
    else:
        # Even turn = Agent B spoke
        current_memory.append(("agent_b", state["result"]))
    
    # Keep only last 2 exchanges (1 from each agent) for fixed memory
    # This ensures agents only see the most recent exchange pair
    if len(current_memory) > 2:
        current_memory = current_memory[-2:]
    
    # Update state with the trimmed memory
    state["memory"] = current_memory
    
    # Increment turn counter
    state["turn"] += 1
    
    # Update round counter (2 turns = 1 round)
    state["round"] = (state["turn"] + 1) // 2
    
    logging.info(f"[Memory Update] Turn {state['turn']-1} completed, Round {state['round']}")
    logging.info(f"[Memory Update] Current memory size: {len(state['memory'])}")
    return state

# Node: Round Checker
def Round_Check(state: AgentState) -> str:
    # Check if we've completed 6 turns (3 rounds, 3 exchanges each agent) - reduced for token limits
    print(f"[Round Check] Current turn: {state['turn']}")
    if state["turn"] > 8:
        print("[Round Check] Moving to Judge")
        return "Judge"
    else:
        print("[Round Check] Continuing debate")
        return "Round_Controller"

# Node: Judge
def Judge(state: AgentState) -> AgentState:
    judge_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a debate judge. Summarize both sides briefly and declare a winner in max 100 words."),
        ("user", "Topic: {topic}\nExchanges: {debate_text}\nWho won and why?")
    ])
    
    llm_judge = Ollama(model = "llama3.2")  # Switched to smaller model
    judge_chain = judge_prompt | llm_judge | parser

    # Create concise summary of exchanges
    debate_text = ""
    memory = state.get("memory", [])
    if memory:
        for i, (agent, msg) in enumerate(memory, 1):
            role_name = state["agent_a"] if agent == "agent_a" else state["agent_b"]
            # Truncate long messages
            short_msg = msg[:100] + "..." if len(msg) > 100 else msg
            debate_text += f"{role_name}: {short_msg} | "
    else:
        debate_text = "No exchanges recorded."

    try:
        state["result"] = judge_chain.invoke({
            "topic": state["topic"],
            "debate_text": debate_text
        })
        logging.info(f"[Judge Node] Verdict: {state['result']}")
        print(f"\n{'='*50}")
        print("JUDGE'S VERDICT:")
        print(f"{'='*50}")
        print(state['result'])
    except Exception as e:
        logging.error(f"[Judge Error] {e}")
        state["result"] = f"Judge cannot provide verdict due to API limits. Debate completed with {len(memory)} exchanges."
        print(f"\n{'='*50}")
        print("DEBATE COMPLETED:")
        print(f"{'='*50}")
        print(state['result'])
    
    return state

# === Build Graph ===
graph = StateGraph(AgentState)

# Add nodes
graph.add_node("Initiate_Node", initiate_node)
graph.add_node("Round_Controller", lambda state: state)  # Pass-through node
graph.add_node("Call_Agent_A", Agent_A)
graph.add_node("Call_Agent_B", Agent_B)
graph.add_node("Memory_Node", Memory_Node)
graph.add_node("Round_Check", lambda state: state)  # Pass-through node
graph.add_node("Judge", Judge)

# Add edges
graph.add_edge(START, "Initiate_Node")
graph.add_edge("Initiate_Node", "Round_Controller")

# Conditional edge from Round_Controller to determine which agent speaks
graph.add_conditional_edges(
    "Round_Controller", 
    round_controller, 
    {
        "Call_Agent_A": "Call_Agent_A",
        "Call_Agent_B": "Call_Agent_B"
    }
)

# Both agents go to Memory_Node after speaking
graph.add_edge("Call_Agent_A", "Memory_Node")
graph.add_edge("Call_Agent_B", "Memory_Node")

# Memory_Node goes to Round_Check
graph.add_edge("Memory_Node", "Round_Check")

# Conditional edge from Round_Check to continue or judge
graph.add_conditional_edges(
    "Round_Check", 
    Round_Check, 
    {
        "Judge": "Judge",
        "Round_Controller": "Round_Controller"
    }
)

# Judge ends the debate
graph.add_edge("Judge", END)

# === Compile and Run ===
app = graph.compile()

# Set recursion limit to prevent infinite loops
app.config = {"recursion_limit": 100}  # Reduced from default 25

# Input configuration
inputs = {
    "topic": "Should AI be regulated like medicine?",
    "agent_a": "Scientist",
    "agent_b": "Philosopher", 
    "round": 1,
    "turn": 1,
    "memory": [],
    "result": ""
}

print("Starting AI Regulation Debate...")
print(f"Topic: {inputs['topic']}")
print(f"Agent A: {inputs['agent_a']}")
print(f"Agent B: {inputs['agent_b']}")
print(f"{'='*50}")

try:
    result = app.invoke(inputs)
    print(f"\n{'='*50}")
    print("DEBATE COMPLETED SUCCESSFULLY")
    print(f"Total turns completed: {result.get('turn', 'Unknown')}")
    print(f"Final memory size: {len(result.get('memory', []))}")
    print(f"{'='*50}")
except Exception as e:
    print(f"Error during debate execution: {e}")
    logging.error(f"Debate execution error: {e}")