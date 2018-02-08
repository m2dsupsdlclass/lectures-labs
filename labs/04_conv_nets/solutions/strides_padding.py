def my_init(shape=(5, 5, 3, 3), dtype=None):
    array = np.zeros(shape=shape)
    array[2, 2] = np.eye(3)
    return array


conv_strides_same = Sequential([
    Conv2D(filters=3, kernel_size=5, strides=2,
           padding="same", kernel_initializer=my_init,
           input_shape=(None, None, 3))
])
conv_strides_valid = Sequential([
    Conv2D(filters=3, kernel_size=5, strides=2,
           padding="valid", kernel_initializer=my_init,
           input_shape=(None, None, 3))
])

img_in = np.expand_dims(sample_image, 0)
img_out_same = conv_strides_same.predict(img_in)
img_out_valid = conv_strides_valid.predict(img_in)

print("Shape of result with SAME padding:", img_out_same.shape)
print("Shape of result with VALID padding:", img_out_valid.shape)

fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, figsize=(12, 4))
ax0.imshow(img_in[0].astype(np.uint8))
ax1.imshow(img_out_same[0].astype(np.uint8))
ax2.imshow(img_out_valid[0].astype(np.uint8))

# We observe that the stride divided the size of the image by 2
# In the case of 'VALID' padding mode, no padding is added, so 
# the size of the ouput image is actually 1 less because of the
# kernel size