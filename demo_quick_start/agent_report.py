import os
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.prompts import PromptTemplate
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.llms import OpenAI
from langchain.agents import AgentType

import glob
import codecs
import chardet


# 监测文件的encoding
def detect_encoding(file_path):
    with open(file_path, 'rb') as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding']
        confidence = result['confidence']
        print(f"File encoding: {encoding}, Confidence: {confidence}")
        return encoding


# 转换文档的编码方式
def convert_encoding(input_file, output_file):
    with codecs.open(input_file, 'r', 'utf-8') as file:
        content = file.read()

    with codecs.open(output_file, 'w', 'gbk') as file:
        file.write(content)


# 加载目录下所有文件
def directory_load(dir_path):
    import os
    documents = list()
    glob_path = dir_path + '/*.txt'
    print('glob_path: ', glob_path)
    for filename in glob.glob(glob_path):
        print(filename)
        if (detect_encoding(filename) == 'utf-8'):
            convert_encoding(filename, filename)
        loader = TextLoader(filename)
        documents.extend(loader.load())
    return documents


# 分隔documents
def split_document(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    texts = text_splitter.split_documents(documents)
    print("len texts: ", len(texts))
    return texts


# 创建embeddings



# 加载持久化向量
def load_embeddings(embeddings, path):
    # 加载数据
    docsearch = Chroma(persist_directory=path, embedding_function=embeddings)
    return docsearch


def create_and_save_embeddings(documents, embeddings, path):
    # embeddings = OpenAIEmbeddings()

    docsearch = Chroma.from_documents(documents, embeddings, persist_directory=path)
    docsearch.persist()
    return docsearch


def run_query(docsearch, query):
    llm = ChatOpenAI(temperature=0, max_tokens=1024)
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


def insert_document(embeddings):
    documents = directory_load("./report")
    print("data has loaded")
    documents = split_document(documents)
    retriever = create_and_save_embeddings(documents, embeddings, path="embeddings/")
    return retriever


def web_search(question):
    # 加载 OpenAI 模型
    os.environ["SERPAPI_API_KEY"] = 'a89ebd9538b3c69783d230411f18f0f315c58531c296dda01a2c0d14cd0bc895'
    llm = OpenAI(temperature=0, max_tokens=2048)

    # 加载 serpapi 工具
    tools = load_tools(["serpapi"])

    # 如果搜索完想在计算一下可以这么写
    # tools = load_tools(['serpapi', 'llm-math'])

    # 如果搜索完想再让他再用python的print做点简单的计算，可以这样写
    # tools=load_tools(["serpapi","python_repl"])

    # 工具加载后都需要初始化，verbose 参数为 True，会打印全部的执行详情
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

    # 运行 agent
    # agent.run("What's the date today? What great events have taken place today in history?")
    # agent.run("Where are we now? What is the weather in my area?")
    agent.run(question)


def main():
    print('begin!')
    embeddings = creat_embeddings()
    docsearch = load_embeddings(embeddings, path="embeddings/")
    print('have load!')
    query = "如何制作可乐？"
    # query = "How much change did Goku learn?(translate to chinese),summerize at 20words"
    response = run_query(docsearch, query)
    print("response:!!!!!!!!!!!!!!!!!!!!!!!!!", response)
    if (response['result'] == '无法根据知识库内容获取相关信息。'):
        web_search(query)


if __name__ == '__main__':
    main()