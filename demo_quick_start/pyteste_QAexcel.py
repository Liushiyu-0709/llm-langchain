# from dotenv import load_dotenv, find_dotenv
import os
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings

os.environ["OPENAI_API_KEY"] = "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"
#env 文件写法 export OPENAI_API_KEY='key'
def load_csv(file_path):
    loader = CSVLoader(file_path=file_path)
    docs = loader.load()
    return docs

def create_embeddings(docs):
    from langchain.embeddings import HuggingFaceEmbeddings

    model_name = 'moka-ai/m3e-base'
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    #embeddings = OpenAIEmbeddings()
    db = DocArrayInMemorySearch.from_documents(
        docs, 
        embeddings
    )
    retriever = db.as_retriever()
    return retriever

def run_query(llm, retriever, query):
    qa_stuff = RetrievalQA.from_chain_type(llm=llm, chain_type="refine", retriever=retriever, verbose=True)
    response = qa_stuff.run(query)
    return response

def main():
    #_ = load_dotenv(find_dotenv())
    file_path = 'OutdoorClothingCatalog_5.csv'
    docs = load_csv(file_path)
    print("have loaded")
    retriever = create_embeddings(docs)
    print("get retriever")
    llm = ChatOpenAI(temperature=0.0, max_tokens=1024)
    query ="Please list all your shirts with sun protection \
    in a table in markdown and summarize each one."
    response = run_query(llm, retriever, query)
    print(response)

if __name__ == '__main__':
    main()