image = tf.placeholder(tf.float32, [None, None, None, 3])
output_image = tf.nn.max_pool(image, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

with tf.Session() as sess:
    feed_dict={image:[sample_image], kernel: kernel_data}
    conv_img = sess.run(output_image, feed_dict=feed_dict)
    print("max pooling output shape:", conv_img.shape)
    show(conv_img[0])

# it is not possible to build a max pooling with a regular convolution
# however it is possible to build average pooling with well  
# chosen strides and kernel
