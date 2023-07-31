from typing import Any

from langchain import OpenAI, PromptTemplate, LLMChain
import os

from langchain.output_parsers import CommaSeparatedListOutputParser, ResponseSchema, StructuredOutputParser
from langchain.tools import BaseTool
from glo import glo_init, get_value, set_value


is_init = False

import json


# 从文件中读取字典
def read_dict_from_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def creat_llm():
    llm = OpenAI(temperature=0, max_tokens=4096 * 2, model_name="gpt-3.5-turbo-16k-0613")
    return llm


class LLMSplit(BaseTool):
    name = "题目分离"
    description = "当你需要将一个问题集分离为多个问题时使用。"
    return_direct = True

    def _run(self, query: str) -> str:
        '''
        response_schemas = [
            ResponseSchema(name="answer", description="answer to the user's question"),
            ResponseSchema(name="source", description="source used to answer the user's question, should be a website.")
        ]
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        '''
        prompt_template_cn = '''
        接下来你将看到一些选择题与简答题，请识别不同的题目并分开，并将多个或一个文本组合为一个 python List 输出,每个元素必须使用双引号包围。 
        
        {input}
        
        输出结果:
        '''
        output_parser = CommaSeparatedListOutputParser()
        prompt = PromptTemplate(template=prompt_template_cn, input_variables=['input'], output_parser=output_parser)
        llm_chain : LLMChain
        if get_value('use_memery'):
            global memory
            llm_chain =LLMChain(prompt=prompt, llm=creat_llm(), memory=memory)
        else:
            llm_chain = LLMChain(prompt=prompt, llm=creat_llm())
        return llm_chain.predict(input=query)

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("BingSearchRun不支持异步")
