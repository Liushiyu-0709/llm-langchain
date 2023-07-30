import glob
import json

from langchain.output_parsers import CommaSeparatedListOutputParser

from agent import creat_agent_split, creat_agent_answer
from llm_tool import init
from vector_tool import make_db
from vector_tool import insert_db
from llm_tool import LLMSplit

agent_split = creat_agent_split()
agent_answer = creat_agent_answer()
# 44,704,712
'''
basic_path = 'report_temp/*.txt'
list = []
for filename in glob.glob(basic_path):
    list.append(filename)
insert_db(list)
print('make db successfully')
'''
init()

query_string = '''在社会主义革命和建设时期，党面临的主要任务是：
    
A. 推动经济发展，提高人民生活水平
B. 实现军事现代化，确保国家安全
C. 实现从新民主主义到社会主义的转变
D. 扩大外交影响力，争取国际地位的提升
自党的十八大以来，中国特色社会主义进入新时代，党面临的主要任务是：

A. 持续推进全面深化改革，全面完善社会主义制度，不断推进国家治理体系和治理能力现代化。
B. 推动经济高质量发展，建设现代化强国，加快实施创新驱动发展战略，提升科技创新能力和核心竞争力，促进经济持续健康发展。
C. 实现第一个百年奋斗目标，开启实现第二个百年奋斗目标新征程，朝着实现中华民族伟大复兴的宏伟目标继续前进。
D. 提升国防实力，强化国家安全体系，坚决维护国家主权、安全、发展利益，推动构建人类命运共同体。
中国共产党百年奋斗的历史意义是什么?
'''.replace('\n', '').replace(' ', '')


def run(query_string):
    query_string.replace('\n', '').replace(' ', '')
    response = LLMSplit().run(query_string)
    split_query = json.loads(response)
    res = []
    for query in split_query:
        temp = (agent_answer.run(query))
        res.append(temp)
    return res


print(run(query_string))
