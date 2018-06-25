from skimage.io import imread
from skimage.transform import resize
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import decode_predictions

path = "images_resize/000007.jpg"

img = imread(path)
plt.imshow(img)

img = resize(img, (224, 224), mode='reflect', preserve_range=True,
             anti_aliasing=True)
# add a dimension for a "batch" of 1 image
img_batch = preprocess_input(img[np.newaxis]) 

predictions = model.predict(img_batch)
decoded_predictions= decode_predictions(predictions)

for s, name, score in decoded_predictions[0]:
    print(name, score)