inp = Input((None, None, 3), dtype="float32")
x = MaxPool2D(2)(inp)

max_pool = Model(inputs=inp, outputs=x)
img_out = max_pool.predict(np.expand_dims(sample_image, 0))
show(img_out[0])


# it is not possible to build a max pooling with a regular convolution
# however it is possible to build average pooling with well  
# chosen strides and kernel
