import os
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.prompts import PromptTemplate
import glob
import codecs
import chardet

os.environ["OPENAI_API_KEY"] = "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"

#监测文件的encoding
def detect_encoding(file_path):
    with open(file_path, 'rb') as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding']
        confidence = result['confidence']
        print(f"File encoding: {encoding}, Confidence: {confidence}")
        return encoding
#转换文档的编码方式
def convert_encoding(input_file, output_file):
    with codecs.open(input_file, 'r', 'utf-8') as file:
        content = file.read()

    with codecs.open(output_file, 'w', 'gbk') as file:
        file.write(content)
#加载目录下所有文件
def directory_load(dir_path):
    import os
    documents = list()
    glob_path = dir_path + '/*.txt'
    print('glob_path: ',glob_path)
    for filename in glob.glob(glob_path):
        print(filename)
        if(detect_encoding(filename) == 'utf-8'):
            convert_encoding(filename, filename)
        loader = TextLoader(filename)
        documents.extend(loader.load())
    return documents
#分隔documents
def split_document(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    texts = text_splitter.split_documents(documents)
    print("len texts: ", len(texts))
    return texts
#创建embeddings
def creat_embeddings():
    from langchain.embeddings import HuggingFaceEmbeddings
    model_name = 'moka-ai/m3e-base'
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    return embeddings

#加载持久化向量
def load_embeddings(embeddings, path):
    # 加载数据
    docsearch = Chroma(persist_directory=path, embedding_function=embeddings)
    return docsearch

def create_and_save_embeddings(documents, embeddings, path):
    #embeddings = OpenAIEmbeddings()

    docsearch = Chroma.from_documents(documents, embeddings, persist_directory=path)
    docsearch.persist()
    return docsearch

def run_query(docsearch, query):
    llm = ChatOpenAI(temperature=0, max_tokens=1024)
    #准备prompt模板
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
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever(), chain_type_kwargs=chain_type_kwargs, return_source_documents=True, verbose=True)
    response = qa_chain({"query": query})
    return response

def insert_document(embeddings):
    documents = directory_load("./report")
    print("data has loaded")
    documents = split_document(documents)
    retriever = create_and_save_embeddings(documents, embeddings, path="embeddings/")
    return retriever

def main():
    print('begin!')
    embeddings = creat_embeddings()
    docsearch = load_embeddings(embeddings, path="embeddings/")
    print('have load!')
    query = "如何制作可乐？"
    #query = "How much change did Goku learn?(translate to chinese),summerize at 20words"
    response = run_query(docsearch, query)
    print("response:!!!!!!!!!!!!!!!!!!!!!!!!!",response)

if __name__ == '__main__':
    main()