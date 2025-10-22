from dotenv import load_dotenv

import logging
import streamlit as st
from langchain_core.messages import HumanMessage

logging.basicConfig(level=logging.INFO)
config = {'configurable': {'thread_id': '1'}}


with st.sidebar:
    st.header('Overview')
    st.caption('This system leverages the latest advancements in machine learning to help you identify, explore, and investigate sensor anomalies.')
    st.caption("Ask any question you'd like -- the assistant is directly connected to the anomaly detection system and can help you explore and understand system-flagged anomalies. It can even plot data for you!")
    st.divider()
    st.header('Links')
    st.link_button('Dataset', 'https://zenodo.org/records/6854240', width=200)
    st.link_button('Source Code', 'https://github.com/tungtngyn', width=200)
    st.divider()
    st.header('Developer')
    st.write('Tung Nguyen')
    st.badge('tnguyen844&#8204;@gatech.edu', icon=':material/mail:', color='gray')
    st.badge('tungtngyn&#8204;@gmail.com', icon=':material/mail:', color='gray')

st.html('<b>AnomalyGPT</b>')

# Load LLM Graph
if 'llm' not in st.session_state:

    # Replace with path to your .env
    # .env file should have OPENAI_API_KEY=sk-...
    load_dotenv(r'../../.env')

    from llm import llm_graph
    st.session_state.llm = llm_graph

# Avatar
role_map = {
    'user': ':material/account_circle:',
    'assistant': ':material/robot_2:'
}

# Initialize Streamlit Messages
if 'messages' not in st.session_state:
    st.session_state.messages = [{'role': 'assistant', 'content': 'What can I help you with today?'}]

# Load Messages
for message in st.session_state.messages:
    with st.chat_message(message['role'], avatar=role_map[message['role']]):
        if message.get('type') == 'image':
            st.image(message['content'])

        else:
            st.markdown(message['content'])

# Check for User Input
if (prompt := st.chat_input('Write your message here..')):

    # Append User Input to Message History
    with st.chat_message('user', avatar=role_map.get('user')):
        st.markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

    # Streaming Function
    input = [HumanMessage(content=prompt)]

    async def stream_tokens():
        async for event in st.session_state.llm.astream_events({'messages': input}, config, stream_mode='messages'):
            if event['event'] == 'on_chat_model_stream':
                yield event['data']['chunk'].content

            elif event['event'] == 'on_tool_end':
                if event['name'] in (
                        'update_analog_sensor_plot_for_user', 
                        'update_digital_sensor_plot_for_user'
                    ):
                    st.session_state.current_img_path = event['data']['output'].content

    # Reset Image Path
    st.session_state.current_img_path = None

    # Stream Text Tokens
    with st.chat_message('assistant', avatar=role_map.get('assistant')):
        full_response = st.write_stream(stream_tokens())
        st.session_state.messages.append({'role': 'assistant', 'content': full_response})
        
    # Plot Images
    if st.session_state.current_img_path:
        with st.chat_message('assistant', avatar=role_map.get('assistant')):
            st.image(st.session_state.current_img_path)
            st.session_state.messages.append({
                'role': 'assistant', 
                'content': st.session_state.current_img_path, 
                'type': 'image'
            })