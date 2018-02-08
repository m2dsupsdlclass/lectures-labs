def my_init(shape, dtype=None):
    array = np.array([
        [0.0,  0.2, 0.0],
        [0.0, -0.2, 0.0],
        [0.0,  0.0, 0.0],
    ])
    # adds two axis to match the required shape (3,3,1,1)
    return np.expand_dims(np.expand_dims(array,-1),-1)



inp = Input((None, None, 1), dtype="float32")
x = Conv2D(kernel_size=(3,3), filters=1,
           padding="same", kernel_initializer=my_init)(inp)

conv_edge = Model(inputs=inp, outputs=x)
img_out = conv_edge.predict(np.expand_dims(grey_sample_image, 0))
show(img_out[0])

# We only showcase a vertical edge detection here.
# Many other kernels work, for example differences
# of centered gaussians (sometimes called mexican-hat
# connectivity)
#
# You may try with this filter as well
# np.array([
#         [ 0.1,  0.2,  0.1],
#         [ 0.0,  0.0,  0.0],
#         [-0.1, -0.2, -0.1],
#     ])
