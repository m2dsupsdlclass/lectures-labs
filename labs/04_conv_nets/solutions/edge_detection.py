image = tf.placeholder(tf.float32, [None, None, None, 1])
kernel = tf.placeholder(tf.float32, [3, 3])

def conv(x, k):
    k = tf.reshape(k, shape=[3, 3, 1, 1])
    return tf.nn.conv2d(x, k, strides=[1, 1, 1, 1],
                        padding='SAME')
    
output_image = conv(image, kernel)

kernel_data = np.array([
        [0.0,  0.2, 0.0],
        [0.0, -0.2, 0.0],
        [0.0,  0.0, 0.0],
    ])
# kernel_data = np.array([
#         [ 0.1,  0.2,  0.1],
#         [ 0.0,  0.0,  0.0],
#         [-0.1, -0.2, -0.1],
#     ])
print(kernel_data)

with tf.Session() as sess:
    feed_dict={image:[grey_sample_image], 
               kernel: kernel_data}
    conv_img = sess.run(output_image, feed_dict=feed_dict)
    print("Resulting image shape:", conv_img.shape)
    show(conv_img[0])

# We only showcase a vertical edge detection here.
# Many other kernels work, for example differences
# of centered gaussians (sometimes called mexican-hat
# connectivity)