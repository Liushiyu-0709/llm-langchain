from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.tools import BaseTool
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from embedding_tool import creat_embeddings
from langchain.agents import Tool
from llm_tool import creat_llm
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI


class VectorTool(BaseTool):
    name = "向量数据库"
    description = "当你需要回答有关中国政治问题时有用"
    dir_path = ''

    def _run(self, query: str) -> str:
        embeddings = creat_embeddings()
        llm = creat_llm()
        # document = load_tool.directory_load(dir_path)
        # document = load_tool.split_document(document)
        # docsearch = create_and_save_vector(document, embeddings, 'db/')

        prompt_template_cn = """我想让你扮演一个知识渊博，逻辑严密的老师，请阅读以下的文本并根据文本内容来对学生提出的问题进行细致的讲解。如果无法根据文本内容获取到信息，请输出’无法根据知识库内容获取相关信息‘，不要尝试去胡编乱造。

            {context}

            学生提出的问题: {question}
            中文回答:
            """
        docsearch = Chroma(persist_directory=self.dir_path, embedding_function=embeddings)
        print('make db successfully')
        prompt_template = PromptTemplate(
            template=prompt_template_cn, input_variables=["context", "question"]
        )
        chain_type_kwargs = {"prompt": prompt_template}
        qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever(),
                                               chain_type_kwargs=chain_type_kwargs, return_source_documents=True,
                                               verbose=True)
        output_parser = CommaSeparatedListOutputParser()
        split_query = output_parser.parse(query)
        res = []
        for query in split_query:
            res.append(qa_chain({"query": query})['result'])
        return '[' + ','.join(res) + ']'

    async def _arun(self, query: str) -> str:
        """异步使用工具。"""
        raise NotImplementedError("BingSearchRun不支持异步")


'''
def vector_prompt_query(query):
    embeddings = creat_embeddings()
    llm = creat_llm()
    docsearch = Chroma(persist_directory=dir_path, embedding_function=embeddings)
    llm = OpenAI(temperature=0, max_tokens=1024)
    # 准备prompt模板
    prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

    {context}

    Question: {question}
    Answer in Chinese:"""

    prompt_template_cn = """我想让你扮演一个知识渊博，逻辑严密的老师，请阅读以下的文本并根据文本内容来对学生提出的问题进行细致的讲解。如果无法根据文本内容获取到信息，请输出’无法根据知识库内容获取相关信息‘，不要尝试去胡编乱造。

    {context}

    学生提出的问题: {question}
    中文回答:
    """
    PROMPT = PromptTemplate(
        template=prompt_template_cn, input_variables=["context", "question"]
    )
    chain_type_kwargs = {"prompt": PROMPT}
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever(),
                                           chain_type_kwargs=chain_type_kwargs, return_source_documents=True,
                                           verbose=True)
    response = qa_chain({"query": query})
    return response
'''


def add_vector_tool(dir_path):
    embeddings = creat_embeddings()
    llm = creat_llm()
    # document = load_tool.directory_load(dir_path)
    # document = load_tool.split_document(document)
    # docsearch = create_and_save_vector(document, embeddings, 'db/')
    print('make db successfully')
    docsearch = Chroma(persist_directory=dir_path, embedding_function=embeddings)
    db = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever())
    return Tool(
        name='vector_tool',
        func=db.run,
        description='useful for when you need to answer questions about the chinese political topics.'
    )


def create_and_save_vector(documents, embeddings, path):
    # embeddings = OpenAIEmbeddings()

    docsearch = Chroma.from_documents(documents, embeddings, persist_directory=path)
    docsearch.persist()
    return docsearch
