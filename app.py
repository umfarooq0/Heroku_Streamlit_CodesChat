from llama_index import StorageContext, load_index_from_storage
import openai
from llama_index.llms import OpenAI
import os
import streamlit as st
from llama_index import ServiceContext
from langchain import LLMChain
from langchain import PromptTemplate
from langchain.chains import ConversationChain
from langchain.agents import ZeroShotAgent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain import SerpAPIWrapper
from langchain.chat_models import ChatOpenAI
from langchain.agents import Tool
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import SequentialChain
from langchain.chains import RetrievalQA
from langchain.agents import ConversationalChatAgent

os.environ["SERPAPI_API_KEY"] = "387c83df077ba5cc8533eac5f7365981e84300455a2cafcc77e9f0d9d6ef9795"
openai_api_key = os.environ.get['OPEN_API_KEY']

st.title("Codes Chat") 

st.write('''   
PLEASE do not enter any confidential information. This uses ChatGPT which stores user inputs. 
We endeavor to use open-source models in the future. 
        
        ''')
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

## Grid Code
service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo-0613"))
storage_context = StorageContext.from_defaults(persist_dir='GC')
index = load_index_from_storage(storage_context, service_context=service_context)


#STC
storage_context_STC = StorageContext.from_defaults(persist_dir='STC')
index_STC = load_index_from_storage(storage_context_STC, service_context=service_context)

#SQSS
storage_context_SQSS = StorageContext.from_defaults(persist_dir='SQSS')
index_SQSS = load_index_from_storage(storage_context_SQSS, service_context=service_context)


#storage_context_CUSC = StorageContext.from_defaults(persist_dir='CUSC')
#index_CUSC = load_index_from_storage(storage_context_CUSC, service_context=service_context)




prefix = """I want you take the role of a senior power system engineer in the GB. You are responding to questions by junior engineers about compliance and regulation. The audience is going to mainly people
who are not technical experts but may have some knowledge of the GB electricity grid. Respond in very simple language, explaining all the technical terms as if the audience doesnt know anything. Provide references"""
suffix = """Begin!"

{chat_history}
Question: {input}
{agent_scratchpad}"""


prompt_template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, please think rationally answer from your own knowledge base

{context}

Question: {question}
"""

prompt = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

search = SerpAPIWrapper()
llm = ChatOpenAI(temperature=0.2,model="gpt-3.5-turbo-0613")

tools = [
    Tool(
        name="GridCodeQA",
        func=lambda q: str(index.as_query_engine().query(q)),
        description="useful for when you want to answer questions about the GridCode.",
        return_direct=True,
    ),
    
    Tool(
        name="STC_QA",
        func= lambda q: str(index_STC.as_query_engine().query(q)),
        description="useful for when you want to answer questions about STC.",
        return_direct=True,
    ),
    Tool(
        name="SQSS_QA",
        func= lambda q: str(index_SQSS.as_query_engine().query(q)),
        description="useful for when you want to answer questions about the SQSS.",
        return_direct=True,
    ),
    Tool(
        name="Intermediate Answer",
        func=search.run,
        description="useful for when you need understand more context around the question or if you are unable to find the answer in the grid Code",
    )
]


prompt = ZeroShotAgent.create_prompt(
    tools,
    prefix=prefix,
    suffix=suffix,
    input_variables=["input", "chat_history", "agent_scratchpad"],
)
memory = ConversationBufferMemory(memory_key="chat_history")
llm_chain = LLMChain(llm=llm, prompt=prompt)


system_msg = """I want you take the role of a senior power system engineer in the GB. You are responding to questions by junior engineers about compliance and regulation. The audience is going to mainly people
who are not technical experts but may have some knowledge of the GB electricity grid. Respond in very simple language, explaining all the technical terms as if the audience doesnt know anything. Provide references"""

agent = ConversationalChatAgent.from_llm_and_tools(
    llm=llm,
    tools=tools,
    system_message=system_msg
)

agent_chain = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True, memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True)
)


#agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
#agent_chain = AgentExecutor.from_agent_and_tools(
#    agent=agent, tools=tools, verbose=True, memory=memory
#)


if prompt := st.chat_input('Ask a question about GB Electricity Codes'):

    openai.api_key = openai_api_key
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    #response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=st.session_state.messages)
    response = agent_chain.run(prompt)

    #msg = response.response
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    with st.chat_message("assistant"):
        st.markdown(response)


