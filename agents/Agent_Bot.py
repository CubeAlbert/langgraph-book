from typing import TypedDict
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

load_dotenv()


llm = ChatOpenAI(model='deepseek-v4-flash')

class AgentState(TypedDict):
    messages: list[HumanMessage]


def chat(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])
    print(response.content)
    return state

graph = StateGraph(AgentState)

graph.add_node("chat", chat)
graph.add_edge(START, "chat")
graph.add_edge("chat", END)

app = graph.compile()


user_input = input("Input: ")
app.invoke(AgentState(messages=[HumanMessage(content = user_input)]))