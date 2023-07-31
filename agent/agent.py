# Import things that are needed generically
"""
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.memory import ConversationBufferMemory

from vector_tool import VectorTool
from llm_tool import creat_llm
from search_tool import creat_web_search_tool
from llm_tool import LLMSplit


def creat_agent_split():
    dir_path = 'report/'
    # add_vector_tool(dir_path), creat_web_search_tool()
    # overall_chain = SimpleSequentialChain(chains=[LLMSplit(), VectorTool()], verbose=True)
    tools = [LLMSplit()]
    memory = ConversationBufferMemory(memory_key="chat_history")
    llm = creat_llm()
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    # agent = initialize_agent(tools, llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, verbose=True,memory=memory)
    return agent


def creat_agent_answer():
    tools = [VectorTool(), creat_web_search_tool()]
    llm = creat_llm()
    agent_instructions = '尽可能多的首先使用向量数据库，使用中文回答'
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, max_iterations=10, )
    return agent
"""