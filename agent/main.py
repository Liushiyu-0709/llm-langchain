import glob

from langchain.output_parsers import CommaSeparatedListOutputParser

from agent import creat_agent_split, creat_agent_answer
from llm_tool import init
from vector_tool import make_db
from vector_tool import insert_db

agent_split = creat_agent_split()
agent_answer = creat_agent_answer()
#44,704,712
basic_path = 'report_temp/*.txt'
list = []
for filename in glob.glob(basic_path):
    list.append(filename)
insert_db(list)
print('make db successfully')
'''
init()

response = agent_split.run(
    '可乐怎么做？'
)
print("res: ", response)
output_parser = CommaSeparatedListOutputParser()
split_query = output_parser.parse(response)
for query in split_query:
    agent_answer.run(query)
'''