import os
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import VectorDBQA
# pip install Chromadb  持久化向量数据库
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import TextLoader
os.environ["OPENAI_API_KEY"] = "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"

import glob
from langchain.document_loaders import TextLoader

def create_embeddings():
    print("enter create_embeddings")
    #loader = DirectoryLoader('./rawdata', glob='*.txt')
    loader = TextLoader('./report/test.txt')
    documents = loader.load()
    print("data has loaded")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=60, chunk_overlap=20)
    texts = text_splitter.split_documents(documents)
    print(len(texts))
    #embeddings = OpenAIEmbeddings()
    from langchain.embeddings import HuggingFaceEmbeddings

    model_name = 'moka-ai/m3e-base'
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    docsearch = DocArrayInMemorySearch.from_documents(texts, embeddings)
    return docsearch

def run_query(vectorstore, query):
    llm = ChatOpenAI(temperature=0, max_tokens=512)
    qa_stuff = VectorDBQA.from_chain_type(llm=llm, chain_type="stuff", k = 3, verbose=True, vectorstore=vectorstore, return_source_documents=True)
    print("qa_stuff: ",qa_stuff)
    response = qa_stuff({"query": query})
    return response

def main():

    retriever = create_embeddings()

    query = "九一八事变后，国内的主要矛盾是什么？"
    #query = "How much change did Goku learn?(translate to chinese),summerize at 20words"
    response = run_query(retriever, query)
    print("response:!!!!!!!!!!!!!!!!!!!!!!!!!",response)

if __name__ == '__main__':
    main()