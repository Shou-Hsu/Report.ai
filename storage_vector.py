from langchain.docstore.document import Document
from langchain.vectorstores import Pinecone
import json, os

def pinecone_storage(file_name:str) -> None:
    from utils import pinecone, embeddings

    with open(f"./summary/{file_name}.json", "r") as f: summary = json.load(f)

    index_name = file_name.lower().replace('_','-')
    # First, check if our index already exists. If it doesn't, we create it
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(
            name=index_name,
            metric='cosine',
            dimension=1536  
    )

    for subtopic in summary['Subtopics']:
        content = ""
        for key, value in subtopic.items():
            if key != 'timestamp':
                content += f'{key}: {"".join(value)}\n'
        doc = Document(page_content=content , metadata={"source": file_name})
        Pinecone.from_documents([doc], embeddings, index_name=index_name)

def chroma_storage(file_name:str, collection_name:str='my-collection', persist_directory:str='chroma_db') -> None:
    from langchain.vectorstores import Chroma
    from utils import embeddings

    with open(f"./summary/{file_name}.json", "r") as f: summary = json.load(f)

    docs = list()
    for subtopic in summary['Subtopics']:
            content = ""
            for key, value in subtopic.items():
                if key != 'timestamp':
                    content += f'{key}: {"".join(value)}\n'
            doc = Document(page_content=content , metadata={"source": file_name})
            docs.append(doc)

    Chroma.from_documents(docs, embeddings, persist_directory=persist_directory, collection_name=collection_name)
