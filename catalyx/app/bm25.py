import math
import re
from collections import defaultdict


def tokenize(text: str) -> list[str]:
    text = text.lower()
    tokens = re.findall(r"[a-z0-9]+", text)
    return tokens


class BM25Index:
    """
    Standard Okapi BM25 implemented from scratch.

    score(D, Q) = sum over query terms t of:
        IDF(t) * ( f(t, D) * (k1 + 1) ) / ( f(t, D) + k1 * (1 - b + b * |D| / avgdl) )

    where:
        f(t, D)  = frequency of term t in document D
        |D|      = length of document D (in tokens)
        avgdl    = average document length across the corpus
        k1, b    = tuning parameters (standard defaults: k1=1.5, b=0.75)
        IDF(t)   = log( (N - n(t) + 0.5) / (n(t) + 0.5) + 1 )
                   N = total number of documents
                   n(t) = number of documents containing term t
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_ids: list[int] = []
        self.doc_tokens: dict[int, list[str]] = {}
        self.doc_lengths: dict[int, int] = {}
        self.avgdl: float = 0.0
        self.doc_freqs: dict[str, int] = defaultdict(int)  # n(t): docs containing term t
        self.N: int = 0
        self._idf_cache: dict[str, float] = {}

    def build(self, documents: dict[int, str]):
        """documents: {product_id: text_to_index}"""
        self.N = len(documents)
        total_length = 0

        for doc_id, text in documents.items():
            tokens = tokenize(text)
            self.doc_ids.append(doc_id)
            self.doc_tokens[doc_id] = tokens
            self.doc_lengths[doc_id] = len(tokens)
            total_length += len(tokens)

            seen_in_doc = set(tokens)
            for term in seen_in_doc:
                self.doc_freqs[term] += 1

        self.avgdl = total_length / self.N if self.N > 0 else 0.0
        self._idf_cache = {}

    def _idf(self, term: str) -> float:
        if term in self._idf_cache:
            return self._idf_cache[term]
        n_t = self.doc_freqs.get(term, 0)
        idf = math.log((self.N - n_t + 0.5) / (n_t + 0.5) + 1)
        self._idf_cache[term] = idf
        return idf

    def _score_doc(self, query_terms: list[str], doc_id: int) -> float:
        tokens = self.doc_tokens[doc_id]
        doc_len = self.doc_lengths[doc_id]
        term_freqs = defaultdict(int)
        for t in tokens:
            term_freqs[t] += 1

        score = 0.0
        for term in query_terms:
            f = term_freqs.get(term, 0)
            if f == 0:
                continue
            idf = self._idf(term)
            numerator = f * (self.k1 + 1)
            denominator = f + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
            score += idf * (numerator / denominator)

        return score

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        query_terms = tokenize(query)
        scores = []
        for doc_id in self.doc_ids:
            score = self._score_doc(query_terms, doc_id)
            if score > 0:
                scores.append((doc_id, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]