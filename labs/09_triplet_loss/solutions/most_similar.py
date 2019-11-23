# Optional: add some test images
more_img_paths = {
    'olivier': [os.path.join('test_images/olivier', img)
                for img in sorted(os.listdir('test_images/olivier'))],
    'charles': [os.path.join('test_images/charles', img)
                for img in sorted(os.listdir('test_images/charles'))],
}
img_paths.update(more_img_paths)
all_images_path = []
for img_list in img_paths.values():
    all_images_path += img_list
path_to_id = {v: k for k, v in enumerate(all_images_path)}
id_to_path = {v: k for k, v in path_to_id.items()}
all_imgs = open_all_images(id_to_path)

# Actually compute the similarities
emb = shared_conv(all_imgs)
emb = emb / np.linalg.norm(emb, axis=-1, keepdims=True)

def most_sim(x, emb, topn=4):
    sims = np.dot(emb, x)
    ids = np.argsort(sims)[::-1]
    return [(id, sims[id]) for id in ids[:topn]]
