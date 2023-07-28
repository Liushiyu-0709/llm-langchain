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
def directory_load(dir_path):
    import os
    documents = list()
    glob_path = dir_path + '/*.txt'
    print('glob_path: ',glob_path)
    for filename in glob.glob(glob_path):
        print(filename, end = ' ')
        loader = TextLoader(filename)
        documents.extend(loader.load())
        print("\n")
    return documents
def create_embeddings():
    documents = directory_load("./report")
    print("data has loaded")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=60, chunk_overlap=20)
    texts = text_splitter.split_documents(documents)
    print("len texts: ",len(texts))
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
    llm = ChatOpenAI(temperature=0.5, max_tokens=1024)
    qa_stuff = VectorDBQA.from_chain_type(llm=llm, chain_type="map_reduce", k = 3, verbose=True, vectorstore=vectorstore, return_source_documents=True)
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