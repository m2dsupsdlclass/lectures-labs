inp = Input((None, None, 3), dtype="float32")
x = AvgPool2D((2,2))(inp)

avg_pool = Model(inputs=inp, outputs=x)
img_out = avg_pool.predict(np.expand_dims(sample_image, 0))
show(img_out[0])

plt.title("avg_pool")
plt.show()

# Same with convolution

def my_init(shape, dtype=None):
    array = np.zeros(shape=(3,3,3,3))
    array[:, :, 0, 0] = 1 / 9.
    array[:, :, 1, 1] = 1 / 9.
    array[:, :, 2, 2] = 1 / 9.
    return array


inp = Input((None, None, 3), dtype="float32")
x2 = Conv2D(kernel_size=(3,3), filters=3,
           padding="same", kernel_initializer=my_init)(inp)

conv_avg = Model(inputs=inp, outputs=x2)
img_out2 = conv_avg.predict(np.expand_dims(sample_image, 0))
show(img_out2[0])
plt.title("conv");

# Note that the numerical computation/approximation might
# be slightly different in the two cases
