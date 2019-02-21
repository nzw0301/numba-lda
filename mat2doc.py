import scipy.io as sio

# large

# data = sio.loadmat('./nips_1-17.mat')
# docs = []
# for i in range(len(data['counts'].indptr)-1):
#     doc = []
#     start_id, last_id = data['counts'].indptr[i], data['counts'].indptr[i+1]
#     for w, freq in zip(
#             data['counts'].indices[start_id:last_id],
#             data['counts'].data[start_id:last_id]):
#
#         for _ in range(freq):
#             doc.append(w)
#
#     docs.append(doc)
#
# with open('nips_1-17.txt', 'w') as f:
#     for doc in docs:
#         f.write(" ".join(map(str, doc)))
#         f.write("\n")

# small

data = sio.loadmat('./nips12raw_str602.mat')
i2w = [w[0][0] for w in data['wl']]
docs = []
for i in range(len(data['counts'].indptr)-1):
    doc = []
    start_id, last_id = data['counts'].indptr[i], data['counts'].indptr[i+1]
    for w, freq in zip(
            data['counts'].indices[start_id:last_id],
            data['counts'].data[start_id:last_id]):

        for _ in range(freq):
            doc.append(i2w[w])
    docs.append(doc)
#
with open('nips12raw_str602.rawtxt', 'w') as f:
    for (i, doc) in enumerate(docs):
        print("\r", i, end="")
        f.write(" ".join(map(str, doc)))
        f.write("\n")
