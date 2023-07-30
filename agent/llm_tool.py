from langchain import OpenAI, PromptTemplate, LLMChain
import os

from langchain.output_parsers import CommaSeparatedListOutputParser, ResponseSchema, StructuredOutputParser
from langchain.tools import BaseTool

is_init = False

import json


# 从文件中读取字典
def read_dict_from_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def init():
    global is_init
    if not is_init:
        config = read_dict_from_file('config.txt')
        os.environ["SERPAPI_API_KEY"] = config["SERPAPI_API_KEY"]
        os.environ["OPENAI_API_KEY"] = config["OPENAI_API_KEY"]
        print(os.environ["SERPAPI_API_KEY"],os.environ["OPENAI_API_KEY"])
        is_init = True


def creat_llm():
    init()
    llm = OpenAI(temperature=0, max_tokens=4096 * 2, model_name="gpt-3.5-turbo-16k")
    return llm


class LLMSplit(BaseTool):
    name = "题目分离"
    description = "当你需要将一个问题集分离为多个问题时使用。"
    return_direct = True

    def _run(self, query: str) -> str:
        response_schemas = [
            ResponseSchema(name="answer", description="answer to the user's question"),
            ResponseSchema(name="source", description="source used to answer the user's question, should be a website.")
        ]
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

        prompt_template_cn = '''
        接下来你将看到一段文本，文本中包含了多个问题，你需要将这段文本按照不同的问题分割成多个文本，并将多个文本组合为 python List 输出。
        
        
        {input}
        
        输出结果:
        '''
        output_parser = CommaSeparatedListOutputParser()
        prompt = PromptTemplate(template=prompt_template_cn, input_variables=['input'], output_parser=output_parser)
        llm_chain = LLMChain(prompt=prompt, llm=creat_llm())
        return llm_chain.predict(input=query)

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("BingSearchRun不支持异步")
