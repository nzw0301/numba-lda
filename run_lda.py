from lda.utils.document import Document
from lda.topic.lda import LDA

import numpy as np

np.random.seed(13)

path = "./nips12raw_str602.rawtxt"
docs = Document().fit(path)

K = 10  # the number of topics
model = LDA(K=K, docs=docs)
model.fit(num_iterations=300)

print()

# phi
for k in range(K):
    print("\ntopic k={}".format(k))
    word_topic_prob = model.word_predict(k)
    for w_id in np.argsort(word_topic_prob)[::-1][:10]:
        print(docs.get_word(w_id), word_topic_prob[w_id])

print()

# theta
for d in range(docs.D):
    print('doc id={}: {}'.format(d, " ".join([docs.get_word(w_id) for w_id in docs.get_document(d)])))
    theta_d = model.topic_predict(d)
    for k, theta_dk in enumerate(theta_d):
        print("topic k={}, θ_dk={}".format(k, theta_dk))
    break
