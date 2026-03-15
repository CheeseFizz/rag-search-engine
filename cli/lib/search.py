from string import punctuation
from nltk.stem import PorterStemmer
from pathlib import Path

import pickle


TOKENIZE_STR_TRANSFORM = str.maketrans("", "", punctuation)

class InvertedIndex:
    def __init__(self, stopwords :list[str]):
        self.index = dict()
        self.docmap = dict()
        self.stopwords = stopwords
        self.__cache = Path(__file__).parent.parent.parent.joinpath("cache")

    def __add_document(self, doc_id, text):
        tdoc = tokenize(text, self.stopwords)
        for t in tdoc:
            if not t in self.index:
                self.index[t] = set()
            self.index[t].add(doc_id)            

    def get_documents(self, term :str) -> list[int]:
        """
        Returns a sorted list of document IDs containing the token, term.
        """
        if not term.lower() in self.index:
            return list()
        return sorted(self.index[term.lower()])

    def build(self, movies :list[dict[str, str]]):
        """
        Builds the index and docmap. Recreates these objects when run.
        """
        i = 1
        self.index = dict()
        self.docmap = dict()
        for m in movies:
            self.docmap[i] = m
            self.__add_document(i, f"{m['title']} {m['description']}")
            i += 1

    def save(self):
        self.__cache.mkdir(parents=True, exist_ok=True)
        with self.__cache.joinpath("index.pkl").open("wb") as f:
            pickle.dump(self.index, f)
        with self.__cache.joinpath("docmap.pkl").open("wb") as f:
            pickle.dump(self.docmap, f)
    
    def load(self):
        if not self.__cache.joinpath("index.pkl").exists() or not self.__cache.joinpath("docmap.pkl").exists():
            raise FileNotFoundError("saved index not found")
        with self.__cache.joinpath("index.pkl").open("rb") as f:
            self.index = pickle.load(f)
        with self.__cache.joinpath("docmap.pkl").open("rb") as f:
            self.docmap = pickle.load(f)
        

def tokenize(text :str, stopwords :list[str]) -> list[str]:
    """
    Converts the input text into a list of tokens.
    """
    stemmer = PorterStemmer()
    
    
    tokens = str.translate(text.lower(), TOKENIZE_STR_TRANSFORM).split(" ")
    
    for t in tokens:
        if t in stopwords or t == "":
            tokens.remove(t)
            continue
        tokens[tokens.index(t)] = stemmer.stem(t)
    return tokens


def keyword_search(query :str, database :InvertedIndex, stopwords :list[str], max_items :int=5) -> list[dict]:
    results = []
    if len(database.index) == 0:
        raise ValueError("database has no data")
    
    tquery = tokenize(query, stopwords)
    
    for t in tquery:
        results += database.get_documents(t)
        if len(results) > max_items:
            results = results[0:max_items]
            break
        
    return results

