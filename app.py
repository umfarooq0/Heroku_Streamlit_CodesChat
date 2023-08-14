from llama_index import StorageContext, load_index_from_storage
import openai
from llama_index.llms import OpenAI
import os
import streamlit as st
from llama_index import ServiceContext


openai_api_key = "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"  
openai.api_key = os.environ["OPENAI_API_KEY"]

st.title("Codes Chat") 

st.write('''💬 Chatbot
This chatbot aims to use ChatGPT to provide answers to questions related to the GB Grid Code.
While ChatGPT has not been 'trained' on the GB Elctricity Codes, it is focused on answering questions on it.
        
PLEASE do not enter any confidential information. This uses ChatGPT which stores user inputs. 
We endeavor to use open-source models in the future. 
        
        ''')
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo-0613"))

storage_context = StorageContext.from_defaults(persist_dir='GC')

index = load_index_from_storage(storage_context, service_context=service_context)

chat_engine = index.as_chat_engine(chat_mode="openai", verbose=True)
#response = chat_engine.chat("reactive power requirements for connections to GB grid")



if prompt := st.chat_input('Ask a question about GB Electricity Codes'):

    openai.api_key = openai_api_key
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    #response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=st.session_state.messages)
    response = chat_engine.chat(prompt,function_call="query_engine_tool")

    msg = response.response
    st.session_state.messages.append({"role": "assistant", "content": msg})
    
    with st.chat_message("assistant"):
        st.markdown(msg)


