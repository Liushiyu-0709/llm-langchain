from langchain.output_parsers import CommaSeparatedListOutputParser

from agent import creat_agent_split, creat_agent_answer
from llm_tool import init
from vector_tool import make_db

agent_split = creat_agent_split()
agent_answer = creat_agent_answer()
'''
make_db()
print('make db successfully')
'''
init()


response = agent_split.run(
    '可乐怎么做？'
)
print("res: ",response)
output_parser = CommaSeparatedListOutputParser()
split_query = output_parser.parse(response)
for query in split_query:
    agent_answer.run(query)
