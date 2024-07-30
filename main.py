from typing import List, Sequence

from dotenv import load_dotenv
load_dotenv()

# Proceed with the rest of the script
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, MessageGraph

from chains import generate_chain, reflect_chain

REFLECT = "reflect"
GENERATE = "generate"

def generation_node(state: Sequence[BaseMessage]):
    return generate_chain.invoke({"messages": state})

def reflection_node(messages: Sequence[BaseMessage]) -> List[BaseMessage]:
    res = reflect_chain.invoke({"messages": messages})
    return [HumanMessage(content=res.content)]

builder = MessageGraph()
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)
builder.set_entry_point(GENERATE)

def should_continue(state: List[BaseMessage]):
    if len(state) > 6:
        return END
    return REFLECT

builder.add_conditional_edges(GENERATE, should_continue)
builder.add_edge(REFLECT, GENERATE)

graph = builder.compile()
# print(graph.get_graph().draw_mermaid())

if __name__ == '__main__':
    inputs = HumanMessage(content="""Make this tweet better:"
                          @LangchainAI
            - newly Tool calling feature is seriously underrated.
            After a long wait, it's here- making the implementation of agents across different models with function calling.
            Made a video covering their newest blog post
            """)
    response = graph.invoke(inputs)

