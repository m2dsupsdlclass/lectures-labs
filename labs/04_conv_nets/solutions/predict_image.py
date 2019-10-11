from skimage.io import imread
from skimage.transform import resize
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.applications.imagenet_utils import decode_predictions

path = "laptop.jpeg"

img = imread(path)
plt.imshow(img)

img = resize(img, (224, 224), mode='reflect', preserve_range=True,
             anti_aliasing=True)
# add a dimension for a "batch" of 1 image
img_batch = preprocess_input(img[np.newaxis]).astype("float32")

predictions = model(img_batch).numpy()
decoded_predictions= decode_predictions(predictions)

for s, name, score in decoded_predictions[0]:
    print(name, score)
