import os
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import VectorDBQA
# pip install Chromadb  持久化向量数据库
from langchain.document_loaders import DirectoryLoader
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
import glob
import codecs
import chardet

os.environ["OPENAI_API_KEY"] = "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"

def detect_encoding(file_path):
    with open(file_path, 'rb') as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding']
        confidence = result['confidence']
        print(f"File encoding: {encoding}, Confidence: {confidence}")
        return encoding
def convert_encoding(input_file, output_file):
    with codecs.open(input_file, 'r', 'utf-8') as file:
        content = file.read()

    with codecs.open(output_file, 'w', 'gbk') as file:
        file.write(content)
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

def split_document(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=60, chunk_overlap=20)
    texts = text_splitter.split_documents(documents)
    print("len texts: ", len(texts))
    return texts
def creat_embeddings():
    from langchain.embeddings import HuggingFaceEmbeddings
    model_name = 'moka-ai/m3e-base'
    model_kwargs = {'device': 'cuda'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    return embeddings
def load_embeddings(embeddings, path):
    # 加载数据
    docsearch = Chroma(persist_directory=path, embedding_function=embeddings)
    return docsearch
def create_and_save_embeddings(documents, embeddings, path):
    #embeddings = OpenAIEmbeddings()

    docsearch = Chroma.from_documents(documents, embeddings, persist_directory=path)
    docsearch.persist()
    return docsearch

def run_query(vectorstore, query):
    llm = ChatOpenAI(temperature=0.5, max_tokens=512)
    qa_stuff = VectorDBQA.from_chain_type(llm=llm, chain_type="map_reduce", k = 3, verbose=True, vectorstore=vectorstore, return_source_documents=True)
    print("qa_stuff: ",qa_stuff)
    response = qa_stuff({"query": query})
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
    retriever = load_embeddings(embeddings, path="embeddings/")
    print('have load!')
    query = "九一八事变后，国内的主要矛盾是什么？"
    #query = "How much change did Goku learn?(translate to chinese),summerize at 20words"
    response = run_query(retriever, query)
    print("response:!!!!!!!!!!!!!!!!!!!!!!!!!",response)

if __name__ == '__main__':
    main()