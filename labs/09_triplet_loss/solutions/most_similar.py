emb = shared_conv.predict(all_imgs)
emb = emb / np.linalg.norm(emb, axis=-1, keepdims=True)

def most_sim(x, emb, topn=5):
    sims = np.dot(emb,x)
    ids = np.argsort(sims)[::-1]
    return [(id,sims[id]) for id in ids[:topn]]
