from sentence_transformers import SentenceTransformer
from pathlib import Path

import json
import numpy as np

class SemanticSearch:

    def __init__(self):

        # Load the model (downloads automatically the first time)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.documents = None
        self.document_map = dict()
        self.__cache = Path(__file__).parent.parent.parent.joinpath("cache")

    def generate_embedding(self, text):
        if len(text.strip()) == 0:
            raise ValueError("generate_embedding: text input was empty or whitespace")
        embedding = self.model.encode([text])[0]
        return embedding
    
    def build_embeddings(self, documents :list[dict]):
        self.documents = documents
        doclist = []
        for doc in documents:
            self.document_map['id'] = doc
            doclist.append(f"{doc['title']}: {doc['description']}")
        self.embeddings = self.model.encode(doclist, show_progress_bar=True)
        self.__save()
        return self.embeddings
    
    def load_or_create_embeddings(self, documents :list[dict]):
        self.documents = documents
        doclist = []
        for doc in documents:
            self.document_map[doc['id']] = doc
            doclist.append(f"{doc['title']}: {doc['description']}")
        if self.__cache.joinpath("movie_embeddings.npy").exists():
            self.__load()
            if len(self.embeddings) == len(self.documents):
                return self.embeddings
        
        self.build_embeddings(documents)
        return self.embeddings
        

    def __save(self):
        self.__cache.mkdir(parents=True, exist_ok=True)
        np.save(self.__cache.joinpath("movie_embeddings.npy"), self.embeddings)

    def __load(self):
        self.embeddings = np.load(self.__cache.joinpath("movie_embeddings.npy"))



def verify_model():
    ss = SemanticSearch()
    print(f"Model loaded: {ss.model}")
    print(f"Max sequence length: {ss.model.max_seq_length}")

def verify_embeddings():
    ss = SemanticSearch()
    doc = Path(__file__).parent.parent.parent.joinpath("data","movies.json")
    with open(doc, "r") as f:
        documents = json.load(f)
    embeddings = ss.load_or_create_embeddings(documents['movies'])

    print(f"Number of docs:   {len(documents['movies'])}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")


def embed_text(text):
    ss = SemanticSearch()
    em = ss.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {em[:3]}")
    print(f"Dimensions: {em.shape[0]}")

def embed_query_text(query):
    ss = SemanticSearch()
    embedding = ss.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")