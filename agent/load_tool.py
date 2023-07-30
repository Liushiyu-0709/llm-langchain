import glob
import codecs
import chardet
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


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
    print('glob_path: ', glob_path)
    for filename in glob.glob(glob_path):
        print(filename)
        if (detect_encoding(filename) == 'utf-8'):
            convert_encoding(filename, filename)
        loader = TextLoader(filename)
        documents.extend(loader.load())
    return documents


def split_document(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    texts = text_splitter.split_documents(documents)
    print("len texts: ", len(texts))
    return texts
