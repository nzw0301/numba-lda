import numpy as np
from numba import int64, float64, void
from numba import njit

from ..discrete.dirichlet import Dirichlet
from ..utils.document import Document


@njit(void(int64[:], int64[:], float64[:], float64, int64, int64, int64[:], int64[:, :], int64[:]))
def gibbs_sampler(doc, z, alpha, beta, V, k, ndk, nkv, nk):
    pro = np.zeros(k)
    for z_index in range(len(doc)):
        word = doc[z_index]
        z_dn = z[z_index]
        nkv[z_dn, word] -= 1
        ndk[z_dn] -= 1
        nk[z_dn] -= 1

        pre_cumsum = 0.
        for t in range(k):
            pro[t] = pre_cumsum = pre_cumsum + (ndk[t] + alpha[t]) * (nkv[t, word] + beta) / (nk[t] + beta * V)
        t = np.searchsorted(pro, pre_cumsum * np.random.rand())

        z[z_index] = t
        nkv[t, word] += 1
        ndk[t] += 1
        nk[t] += 1


class LDA(object):

    def __init__(self, K: int, docs: Document):
        self.K = K
        self._documents = docs.get_documents()
        self._V = docs.get_nb_vocab()
        self._D = docs.get_nb_docs()
        self._beta = 0.01
        self._doc_dirichlet = Dirichlet(K, 0.1)
        self._nkv = np.zeros((self.K, self._V), dtype=np.int)
        self._ndk = np.zeros((self._D, self.K), dtype=np.int)
        self._nk = np.zeros(self.K, dtype=np.int)
        self._z = []
        self._doc_length_counter = np.bincount(docs.get_doc_lengths())
        self.max_doc_len = np.size(self._doc_length_counter) - 1

    def fit(self, num_iterations=300):
        # init topics
        for doc_id, doc in enumerate(self._documents):
            doc_topic = np.random.randint(self.K, size=len(doc))
            self._z.append(doc_topic)
            for word, topic in zip(doc, doc_topic):
                self._nkv[topic, word] += 1
                self._ndk[doc_id, topic] += 1
                self._nk[topic] += 1

        for i in range(1, num_iterations+1):
            print("\r", i, end="")
            for doc_id in range(self._D):
                doc = self._documents[doc_id]

                gibbs_sampler(doc,
                              self._z[doc_id],
                              self._doc_dirichlet.get_alphas(),
                              self._beta,
                              self._V,
                              self.K,
                              self._ndk[doc_id],
                              self._nkv,
                              self._nk)

        return self._z

    def word_predict(self, topic_id: int):
        return (self._nkv[topic_id, :] + self._beta) / (self._nk[topic_id] + self._V * self._beta)

    def topic_predict(self, doc_id: int):
        p = self._ndk[doc_id, :] + self._doc_dirichlet.get_alphas()
        return p / np.sum(p)
