import glob
import json
import os
from glo import glo_init, get_value, set_value

from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.memory import VectorStoreRetrieverMemory
from langchain.vectorstores import Chroma

from embedding_tool import creat_embeddings
from vector_tool import make_db
from vector_tool import insert_db
from llm_tool import LLMSplit, read_dict_from_file, creat_llm
from vector_tool import VectorTool
from search_tool import creat_web_search_tool

# 44,704,712
'''
basic_path = 'report_temp/*.txt'
list = []
for filename in glob.glob(basic_path):
    list.append(filename)
insert_db(list)
print('make db successfully')
'''


def config_init():
    config = read_dict_from_file('config.txt')
    os.environ["SERPAPI_API_KEY"] = config["SERPAPI_API_KEY"]
    os.environ["OPENAI_API_KEY"] = config["OPENAI_API_KEY"]
    print(os.environ["SERPAPI_API_KEY"], os.environ["OPENAI_API_KEY"])


def var_init():
    set_value("LLM_split", LLMSplit())
    set_value("vector_tool", VectorTool())
    set_value("embeddings", creat_embeddings())
    set_value("know_db", Chroma(persist_directory='know', embedding_function=get_value("embeddings")))
    set_value("llm", creat_llm())
    set_value("memory_db", None)
    set_value("web_tool", creat_web_search_tool())
    set_value("use_memory", False)


def all_init():
    config_init()
    glo_init()
    var_init()


query_strings = '''在社会主义革命和建设时期，党面临的主要任务是：
    
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


def my_run(query_string):
    query_string = query_string.replace('\n', '').replace(' ', '')
    print('in_text: ', query_string)
    response = get_value("LLM_split").run(query_string)
    print('out_text: ', response)
    split_query = json.loads(response)
    res = []
    index = 1
    for query in split_query:
        # temp = (agent_answer.run(query))
        temp = str(index) + '.\n'
        temp += get_value("vector_tool").run(query)
        if '无法根据知识库内容获取相关信息' in temp:
            temp += '\t根据网络搜索结果得到：\n'
            temp += get_value("web_tool").run(query)
        res.append(temp)
        index += 1
    return res

# print(my_run(query_strings))
