from langchain.tools import BaseTool
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

import load_tool
import hashlib
from glo import glo_init, get_value, set_value
from langchain.prompts import PromptTemplate


def make_db():
    embeddings = get_value('embeddings')
    dir_path = 'report/'
    document = load_tool.directory_load(dir_path)
    document = load_tool.split_document(document)
    create_and_save_vector(document, embeddings, 'know')


def insert_db(dir_path):
    documents = []
    for document in dir_path:
        load_tool.document_load(document, documents)
        documents = load_tool.split_document(documents)
    create_and_save_vector(documents, get_value('embeddings'), 'know')


class VectorTool(BaseTool):
    name = "向量数据库"
    description = "当你需要回答问题时要用，一定要首先使用此工具。"

    def _run(self, query: str) -> str:
        llm = get_value('llm')
        '''
        prompt_template_cn = """我想让你扮演一个知识渊博，逻辑严密的老师，请阅读以下的文本并根据文本内容来对学生提出的问题进行细致的讲解。如果无法根据文本内容获取到信息，请输出’无法根据知识库内容获取相关信息‘，不要尝试去胡编乱造。

            {context}

            学生提出的问题: {question}
            
            Please answer in Chinese！用中文回答！Veuillez répondre en chinois！Antworten sie bitte auf chinesisch！Пожалуйста, ответьте по-китайски！
            """
        '''
        prompt_template_cn = """
请以一致的格式回答

【问题】构建新发展格局，是以习近平同志为核心的党中央积极应对国际国内形势变化，与时俱进提升我国经济发展水平，塑造国际经济全球和竞争新优势提出的战略决策，这一发展格局是
A.以体制机制创新为主体，利用好国际国内两个市场
B.以维护和平稳定为主体，促进国际国内经济复苏
C.以国内大循环为主体，国内国际双循环相互促进
D.以发展先进制造业为主体，促进产业结构优化升级
【答案】C
【解析】新发展格局是以国内大循环为主体、国内国际双循环相互促进的发展格局，绝不是封闭的国内循环，而是更加开放的国内国际双循环。故正确答案为 C。


【问题】从人民生活水平不断提高的视角，分析我国从 “解决温饱”到“达到小康水平”再到“全面达成小康社会”中体现的中国特色社会主义制度优势。(6 分)
【答案】
第一，消除贫困、改善民生、逐步实现共同富裕，是中国特色社会主义的本质要求。让广大人民群众共享改革发展成果，是社会主义本质要求，是社会主义制度优越性的集中体现。我们推动经济社会发展，归根结底是要实现全体人民共同富裕。
第二，党的集中统一领导的优势。中国共产党的领导是中国特色社会主义制度最大的优势，中国共产党的领导是中国特色社会主义最本质的特征，党的领导为全面建成小康社会提供了根本保障。
第三，坚持以人民为中心的优势。我们的国家是人民当家作主的社会主义国家，坚持以人民为中心的发展思想，始终坚持把人民对美好生活的向往作为自己的奋斗目标。

下面你将回答这个问题,提供给你知识信息: {context}
对于选择题，要回答【答案】和【解析】
问题与知识信息无关，请输出’无法根据知识库内容获取相关信息‘，不要尝试去胡编乱造。
【问题】{question}
"""
        docsearch = get_value("know_db")
        print('make db successfully')
        prompt_template = PromptTemplate(
            template=prompt_template_cn, input_variables=["context", "question"]
        )
        memory = get_value("memory_db")
        chain_type_kwargs = {"prompt": prompt_template}
        qa_chain:RetrievalQA.from_chain_type
        if get_value('use_memery'):
            qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever(),
                                                   chain_type_kwargs=chain_type_kwargs, return_source_documents=True,
                                                   verbose=True, memory=memory)
        else:
            qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever(),
                                               chain_type_kwargs=chain_type_kwargs, return_source_documents=True,
                                               verbose=True)
        res = qa_chain({"query": query})
        print(res['source_documents'])
        return res['result']

    async def _arun(self, query: str) -> str:
        """异步使用工具。"""
        raise NotImplementedError("BingSearchRun不支持异步")


def create_and_save_vector(documents, embeddings, path):
    ids = []
    hash_object = hashlib.sha256()
    for content in documents:
        hash_object.update(content.page_content.encode('utf-8'))
        ids.append(hash_object.hexdigest())
    docsearch = Chroma.from_documents(documents, embeddings, persist_directory=path, ids=ids)
    docsearch.persist()
    docsearch = Chroma(persist_directory=path, embedding_function=get_value('embeddings'))
    set_value(path + "_db", docsearch)
    return docsearch
