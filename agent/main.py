from agent import creat_agent_split, creat_agent_answer
from llm_tool import init

agent_split = creat_agent_split()
agent_answer = creat_agent_answer()

init()

response = agent_split.run(
    '''下列哪个是中国共产党的最高领导机关？ A. 国家主席 B. 全国人民代表大会 C. 中国共产党中央委员会 D. 国务院。中国的国家权力机关是由下列哪个机构组成？ A. 中共中央纪律检查委员会 B. 全国人民代表大会 C. 中国人民政治协商会议 D. 中国国务院。中国的县级行政区划下面一级的行政区是？ A. 乡级行政区 B. 省级行政区 C. 地级行政区 D. 市级行政区。中国共产党的宗旨是实现共产主义社会。（正确/错误）。请简要描述中国共产党的历史发展和目前的领导结构。'''
)
print("res: ",response)
agent_answer.run(response)
