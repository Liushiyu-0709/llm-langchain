from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.tools import BaseTool
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

import load_tool
import hashlib
from embedding_tool import creat_embeddings
from llm_tool import creat_llm
from langchain.prompts import PromptTemplate

docsearch = Chroma(persist_directory='db/', embedding_function=creat_embeddings())


def make_db():
    embeddings = creat_embeddings()
    dir_path = 'report/'
    document = load_tool.directory_load(dir_path)
    document = load_tool.split_document(document)
    create_and_save_vector(document, embeddings, 'db/')


def insert_db(dir_path):
    documents = []
    for document in dir_path:
        load_tool.document_load(document, documents)
        documents = load_tool.split_document(documents)
    create_and_save_vector(documents, creat_embeddings(), 'db/')


class VectorTool(BaseTool):
    name = "向量数据库"
    description = "当你需要回答问题时要用，一定要首先使用此工具。"

    def _run(self, query: str) -> str:
        llm = creat_llm()

        prompt_template_cn = """我想让你扮演一个知识渊博，逻辑严密的老师，请阅读以下的文本并根据文本内容来对学生提出的问题进行细致的讲解。如果无法根据文本内容获取到信息，请输出’无法根据知识库内容获取相关信息‘，不要尝试去胡编乱造。

            {context}

            学生提出的问题: {question}
            
            Please answer in Chinese！用中文回答！Veuillez répondre en chinois！Antworten sie bitte auf chinesisch！Пожалуйста, ответьте по-китайски！
            """
        global docsearch
        print('make db successfully')
        prompt_template = PromptTemplate(
            template=prompt_template_cn, input_variables=["context", "question"]
        )
        chain_type_kwargs = {"prompt": prompt_template}
        qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever(),
                                               chain_type_kwargs=chain_type_kwargs, return_source_documents=True,
                                               verbose=True)

        return qa_chain({"query": query})['result']

    async def _arun(self, query: str) -> str:
        """异步使用工具。"""
        raise NotImplementedError("BingSearchRun不支持异步")


def create_and_save_vector(documents, embeddings, path):
    ids = []
    hash_object = hashlib.sha256()
    for content in documents:
        hash_object.update(content.page_content.encode('utf-8'))
        ids.append(hash_object.hexdigest())

    global docsearch
    docsearch = Chroma.from_documents(documents, embeddings, persist_directory=path, ids=ids)
    docsearch.persist()
    return docsearch
