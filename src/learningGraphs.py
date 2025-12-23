from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages

class State(TypedDict):
    count: int
    tripled_count: int
    messages: Annotated[list[HumanMessage | AIMessage], add_messages]

# Node 1: Process the user's message
def process_message(state: State):
    last_message = state["messages"][-1].content
    return {
        "messages": [AIMessage(content=f"You said: '{last_message}'. This is call #{state['count']}. \
                               \nAnd here is the triple counter: {state['tripled_count']}.")]
    }

# Node 2: Increment the counter
def increment_counter(state: State):
    return {"count": state["count"] + 1}

# Node 3: triple the count
def triple_counter(state: State):
    return {"tripled_count": state["count"] * 3}  # Don't modify state at all

# Build graph
builder = StateGraph(State)

builder.add_node("process", process_message)
builder.add_node("increment", increment_counter)
builder.add_node("triple", triple_counter)

builder.set_entry_point("increment")
builder.add_edge("increment", "triple")
builder.add_edge("triple", "process")
#builder.set_finish_point("increment")
builder.set_finish_point("process")

graph = builder.compile()

# ---- Run ----
result = graph.invoke({
    "count": 0,
    "messages": [HumanMessage(content="Hello")]
})

last_message = result["messages"][-1]

print(f"Count: {result['count']}")
print(f"{type(result['messages'][-1]).__name__}: {result['messages'][-1].content}")

# Run again with new message
result["messages"].append(HumanMessage(content="How are you?"))
result = graph.invoke(result)
print(f"\nCount: {result['count']}")
print(f"{type(result['messages'][-1]).__name__}: {result['messages'][-1].content}")


# Run again with new message
result["messages"].append(HumanMessage(content="I am great!"))
result = graph.invoke(result)
print(f"\nCount: {result['count']}")
print(f"{type(result['messages'][-1]).__name__}: {result['messages'][-1].content}")


print(f"\n\n------------\nCHAT HISTORY\n------------\n\n")

for m in result["messages"]:
    print(f"{type(m).__name__}: {m.content}\n")
