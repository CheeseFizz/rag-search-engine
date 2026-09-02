from sentence_transformers import SentenceTransformer
from pathlib import Path

import json
import numpy as np
import re

class SemanticSearch:

    def __init__(self, model='all-MiniLM-L6-v2'):

        # Load the model (downloads automatically the first time)
        self.embeddings: np.ndarray | None = None
        self.documents: list[dict] | None = None
        self.document_map = dict()
        self._cache = Path(__file__).parent.parent.parent.joinpath("cache")
        self.model = SentenceTransformer(model)

    def generate_embedding(self, text: str) -> object:
        if len(text.strip()) == 0:
            raise ValueError("generate_embedding: text input was empty or whitespace")
        embedding = self.model.encode([text])[0]
        return embedding
    
    def build_embeddings(self, documents :list[dict]):
        self.documents = documents
        doclist = []
        for doc in documents:
            self.document_map[doc['id']] = doc
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
        if self._cache.joinpath("movie_embeddings.npy").exists():
            print("loading embeddings from cache")
            self.__load()
            if len(self.embeddings) == len(self.documents):
                return self.embeddings
        
        print("building embeddings")
        self.build_embeddings(documents)
        return self.embeddings
    
    def search(self, query, limit):
        if self.build_embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        embedded_query: np.ndarray = self.generate_embedding(query)
        sscores = []
        for i, doc_em in enumerate(self.embeddings):
            cs = cosine_similarity(embedded_query, doc_em)
            doc = self.document_map[i+1]
            sscores.append((cs, doc))
        sorted_sscores = sorted(sscores, key=lambda item: item[0], reverse=True)
        results = []
        for i in range(limit):
            d = dict()
            d["score"] = sorted_sscores[i][0]
            d["title"] = sorted_sscores[i][1]["title"]
            d["description"] = sorted_sscores[i][1]["description"]
            results.append(d)
        return results
        

        

    def __save(self):
        self._cache.mkdir(parents=True, exist_ok=True)
        np.save(self._cache.joinpath("movie_embeddings.npy"), self.embeddings)

    def __load(self):
        self.embeddings = np.load(self._cache.joinpath("movie_embeddings.npy"))

class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings: np.ndarray | None = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        doclist: list[str] = []
        all_chunks: list[str] = []
        chunk_md: list[dict] = []
        for i_doc, doc in enumerate(documents):
            self.document_map[doc['id']] = doc
            doclist.append(f"{doc['title']}: {doc['description']}")
            if doc['description'] == "":
                continue
            chunk_temp = semantic_chunk(doc['description'], max_chunk_size=4, overlap=1)
            for i_chunk, tc in enumerate(chunk_temp['chunks']):
                all_chunks.append(tc)
                chunk_md.append(
                    {
                        'movie_idx': i_doc,
                        'chunk_idx': i_chunk,
                        'total_chunks': chunk_temp['numchunks']
                    }
                )
        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = chunk_md


        self._cache.mkdir(parents=True, exist_ok=True)
        np.save(self._cache.joinpath("chunk_embeddings.npy"), self.chunk_embeddings)

        with open(self._cache.joinpath("chunk_metadata.json"), "w") as f:
            json.dump({"chunks": chunk_md, "total_chunks": len(all_chunks)}, f, indent=2)
        
        return self.chunk_embeddings
    
    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        if self._cache.joinpath("chunk_embeddings.npy").exists() and self._cache.joinpath("chunk_metadata.json").exists():
            self.chunk_embeddings = np.load(self._cache.joinpath("chunk_embeddings.npy"))
            with open(self._cache.joinpath("chunk_metadata.json"), "r") as f:
                chunk_md_json = json.load(f)
            self.chunk_metadata = chunk_md_json['chunks']
            self.documents = documents
            doclist: list[str] = []
            for doc in documents:
                self.document_map[doc['id']] = doc
            return self.chunk_embeddings
        
        #if nothing to load, create new
        return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = 10):
        chunk_scores: list[dict] = []
        embeded_query = super().generate_embedding(query)
        for i_chunk, cem in enumerate(self.chunk_embeddings):
            cs = cosine_similarity(cem, embeded_query)
            md = self.chunk_metadata[i_chunk]
            if md['chunk_idx'] != i_chunk:
                raise Exception(f"expected chunk_idx {i_chunk}; got chunk_idx {md['chunk_idx']}")
            chunk_scores.append(
                {
                    'chunk_idx': i_chunk,
                    'movie_idx': md['movie_idx'],
                    'score': cs
                }
            )
        high_scores: dict[int,int] = dict()
        for cs in chunk_scores:
            if cs['movie_idx'] not in high_scores.keys():
                high_scores[cs['movie_idx']] = cs['score']
                continue
            if cs['score'] > high_scores[cs['movie_idx']]:
                high_scores[cs['movie_idx']] = cs['score']

        sorted_scores = sorted(high_scores.items(), key=lambda item: item[1], reverse=True)
        results = []
        for i in range(limit):
            
            d = {
                "id": self.documents[sorted_scores[i][0]]['id'],
                "title": self.documents[sorted_scores[i][0]]['title'],
                "document": self.documents[sorted_scores[i][0]]['description'][:100],
                "score": round(sorted_scores[i][1], SCORE_PRECISION),
                "metadata": self. or {},
            }
            results.append(d)
        return results



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

def chunk(text: str, chunk_size: int, overlap: int) -> dict[str, object]:
    resd = dict()
    resd['numchar'] = len(text)
    resd['lines'] = []

    if overlap >= len(text):
        raise ValueError("overlap exceeds text length")
    if overlap >= chunk_size:
        raise ValueError("overlap exceeds chunk size")

    working = text.split()
    n = 0
    done = False
    while(n < len(working)):
        if n - overlap < 0:
            n = 0
        else:
            n = n - overlap
        if len(working) - chunk_size >= n:
            l = working[n:n+chunk_size]
            n = n + chunk_size
        else:
            l = working[n::]
            n = len(working)
            done = True
        resd['lines'].append(" ".join(l))
        if done:
            break

    return resd
        
def semantic_chunk(text: str, max_chunk_size: int, overlap: int) -> dict[str, object]:
    pattern = r"(?<=[.!?])\s+"
    sentences = re.split(pattern, text)
    resd = dict()
    resd['numchar'] = len(text)
    resd['numchunks'] = 0
    resd['chunks'] = []

    if overlap >= len(sentences):
        raise ValueError("overlap exceeds text length")
    if overlap >= max_chunk_size:
        raise ValueError("overlap exceeds chunk size")

    n = 0
    done = False
    while(n < len(sentences)):
        if n - overlap < 0:
            n = 0
        else:
            n = n - overlap
        if len(sentences) - max_chunk_size >= n:
            l = sentences[n:n+max_chunk_size]
            n = n + max_chunk_size
        else:
            l = sentences[n::]
            n = len(sentences)
            done = True
        resd['chunks'].append(" ".join(l))
        if done:
            resd['numchunks'] = len(resd['chunks'])
            break

    return resd

