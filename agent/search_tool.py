from llm_tool import creat_llm
from langchain.agents import load_tools
def creat_web_search_tool():
    # 加载 OpenAI 模型
    llm = creat_llm()

    # 加载 serpapi 工具
    tools = load_tools(["serpapi"])
    tools[0].description = '当你无法根据知识库信息获取相关信息时调用'
    return tools[0]
