from typing import List


from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import MessagesState
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.documents.base import Document
from utils import (
    query_anomalies,
    query_analog_sensor_importances,
    query_digital_sensor_activations,
    update_analog_sensor_plot_for_user,
    update_digital_sensor_plot_for_user
)

# Load Vector DB
embeddings = OpenAIEmbeddings(model='text-embedding-3-large')

vector_db = Chroma(
    collection_name='train_metadata',
    persist_directory=r'../02-data/chroma_db',
    embedding_function=embeddings
)

# Initialize LLM
tools = [
    query_anomalies,
    query_analog_sensor_importances,
    query_digital_sensor_activations,
    update_analog_sensor_plot_for_user,
    update_digital_sensor_plot_for_user
]
llm = ChatOpenAI(model='gpt-4o', streaming=True).bind_tools(tools, parallel_tool_calls=True)

# System Message
sys_msg = SystemMessage(content="""
    You are a helpful analyst tasked with aiding users understand anomaly detection data.
                        
    Users are non-technical and responses should be clear, concise, and jargon-free.
    
    The data comes from a train that is monitored by 8 analog sensors and 8 digital sensors. 
    The analog sensors are: tp2, tp3, h1, dv_pressure, reservoirs, oil_temperature, flowmeter, motor_current
    The digital sensors are: comp, dv_electric, towers, mpg, lps, pressure_switch, oil_level, caudal_impulses
    
    Sensors record data once per second from 6AM to 2AM the next day each day.
                        
    The user will ask you questions about anomalies that the system has detected. 
    You have access to tools that can help you answer the user's questions. These tools include:
    1. query_anomalies(start_ts: str, end_ts: str) -> Returns a list of anomalies between start_ts and end_ts.
    2. query_analog_sensor_importances(start_ts: str, end_ts: str) -> Returns a dictionary of sensor importances (in seconds out of range) between start_ts and end_ts.
    3. query_digital_sensor_activations(start_ts: str, end_ts: str) -> Returns a dictionary of digital sensor activations (in seconds activated) between start_ts and end_ts.
    4. update_analog_sensor_plot_for_user(sensor_name: str, start_ts: str, end_ts: str) -> Plots analog sensor data for the user.
    5. update_digital_sensor_plot_for_user(sensor_name: str, start_ts: str, end_ts: str) -> Plots digital sensor data for the user.
                        
    start_ts and end_ts must be between 2022-01-01 and 2022-06-02 and be in 'YYYY-MM-DD HH:MM:SS' (24H - Military Time) format.
    
    When plotting anomaly events, make sure to always plot 3 hours before and after the event to provide context.
    
    There is only space for one plot per response, do not try to plot multiple sensors as part of the same response.
    Do not use Markdown to render images, the system will render the image automatically after the response is completed.

    The underlying anomaly detection model works by fitting 1 Prophet Forecasting Model per analog sensor.
    For each timestamp, if >5 analog sensors are out of their expected range for >5 minutes, then the timestamp is flagged as an anomaly.
    Digital sensors are not used in the anomaly detection model, but may be useful for understanding anomalies.
    
    Use these tools to gather data and generate plots to help the user understand the anomalies.
""")

# Context Logging
def _write_context_to_file(message: HumanMessage, context: List[Document]) -> None:
    with open(r'./context.log', r'a') as f:
        f.write(f'Prompt ID: {message.id}\n')
        f.write(f'Prompt: {message.content}\n\n')
        f.write(f'Retrieved: {len(context)}, Sample (Top-1):\n')
        f.write(f'{context[0].page_content}\n\n\n')
    return None

# Custom State
class ContextState(MessagesState):
    context: str

# RAG Component 
def retrieve(state: ContextState):
    """Retrieval Augmented Generation (RAG) - Retrieves relevant documents from Vector DB
    based on the latest user prompt and appends them to the conversation history.
    """
    prompt = state['messages'][-1].content
    retrieved_docs = vector_db.similarity_search(prompt, k = 4)
    _write_context_to_file(state['messages'][-1], retrieved_docs)
    return {'context': retrieved_docs}

# Generation Component
def assistant(state: ContextState):
    context_msg = SystemMessage(content=f"""
    The following context was retrieved from the knowledge base to help you answer the user's question:
    {r'\n\n'.join([doc.page_content for doc in state['context']])}
    """)
    return {'messages': [llm.invoke([sys_msg] + [context_msg] + state['messages'])]}

# Build Graph
builder = StateGraph(ContextState)

builder.add_node('retrieve', retrieve)
builder.add_node('assistant', assistant)
builder.add_node('tools', ToolNode(tools))

builder.add_edge(START, 'retrieve')
builder.add_edge('retrieve', 'assistant')
builder.add_conditional_edges(
    'assistant',
    tools_condition,
)
builder.add_edge('tools', 'assistant')

# In-Memory Checkpointer for Demo
memory = MemorySaver()
llm_graph = builder.compile(checkpointer=memory)