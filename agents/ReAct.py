from typing import Annotated, Sequence, TypedDict, cast
from langchain_core.messages import BaseMessage,HumanMessage,AIMessage
from langchain_core.messages import ToolMessage
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv


load_dotenv()


@tool
def add(a:int, b:int) -> int:
    """return the addition of two numbers"""
    return a+b;

@tool
def sub(a:int, b:int) -> int:
    """return the subtraction of two numbers"""
    return a-b;

@tool
def mul(a:int, b:int) -> int:
    """return the multiplication of two numbers"""
    return a*b;

tools = [add, sub, mul]

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

llm = ChatOpenAI(model = "deepseek-v4-flash", use_responses_api= True).bind_tools(tools)


def call_llm(state: AgentState) -> AgentState:
    system_message = SystemMessage(content = "You are a helpful assistant.")
    response = llm.invoke([system_message] + state["messages"])
    #print(response)
    return {"messages": [response]}

def should_exit(state: AgentState):
    last_message = cast(AIMessage, state["messages"][-1])
    if last_message.tool_calls:
        return "continue"
    else:
        return 'exit'


graph = StateGraph(AgentState)

graph.add_node("tools",ToolNode(tools))
graph.add_node("call_llm", call_llm)


graph.add_edge(START, "call_llm")
graph.add_edge("tools", "call_llm")
graph.add_conditional_edges("call_llm", should_exit, {
    "continue":"tools",
    "exit":END
})

app = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

print_stream(app.stream(AgentState(messages=[("user","What's the result of two plus three, then multiple four; then tell me a joke")]), stream_mode="values"))