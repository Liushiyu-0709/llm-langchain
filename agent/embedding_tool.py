from langchain.embeddings import HuggingFaceEmbeddings


def creat_embeddings():
    model_name = 'moka-ai/m3e-base'
    model_kwargs = {'device': 'cuda'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    return embeddings
