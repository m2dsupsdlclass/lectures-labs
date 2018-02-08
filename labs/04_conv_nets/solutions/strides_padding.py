
def my_init(shape, dtype=None):
    array = np.zeros(shape=(5,5,3,3))
    array[2,2] = np.eye(3) 
    return array

inp = Input((None, None, 3), dtype="float32")
x = Conv2D(kernel_size=(5,5), filters=3, strides=2,
           padding="same", kernel_initializer=my_init)(inp)

conv_strides_same = Model(inputs=inp, outputs=x)
x2 = Conv2D(kernel_size=(5,5), filters=3, strides=2,
           padding="valid", kernel_initializer=my_init)(inp)

conv_strides_valid = Model(inputs=inp, outputs=x2)

img_out = conv_strides_same.predict(np.expand_dims(sample_image, 0))
img_out2 = conv_strides_valid.predict(np.expand_dims(sample_image, 0))
show(img_out[0])

print("Shape of result with SAME padding:", img_out.shape)
print("Shape of result with VALID padding:", img_out2.shape)


# We observe that the stride divided the size of the image by 2
# In the case of 'VALID' padding mode, no padding is added, so 
# the size of the ouput image is actually 1 less because of the
# kernel size
