from langchain.embeddings import HuggingFaceEmbeddings

model_name = 'moka-ai/m3e-base'
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
hf = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

embeddings = hf.embed_documents(
    [
        "你好",
        "真不错",
        "What's your name?",
        "My friends call me World",
        "Hello World!"
    ]
)
print(len(embeddings), len(embeddings[0]))

