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
            print("loading embeddings from cache")
            self.__load()
            if len(self.embeddings) == len(self.documents):
                return self.embeddings
        
        print("building embeddings")
        self.build_embeddings(documents)
        return self.embeddings
    
    def search(self, query, limit):

        pass

        if self.build_embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        embedded_query = self.generate_embedding(query)
        sscores = []
        for i, doc_em in enumerate(self.embeddings):
            cs = cosine_similarity(embedded_query, doc_em)
            doc = self.document_map[str(i+1)]
            sscores.append((cs, doc))
        sorted_sscores = sorted(sscores, key=lambda item: item[0], reverse=True)
        results = []
        for i in range(limit):
            d = dict()
            d["score"] = sorted_sscores[i][0]
            d["title"] = sorted_sscores[i][1]["title"]
            d["description"] = sorted_sscores[i][1]["description"]
            results.append(d)
        

        

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

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)
