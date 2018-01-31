image = tf.placeholder(tf.float32, [None, None, None, 3])
kernel = tf.placeholder(tf.float32, [3, 3, 3, 3])

def conv(x, k):
    return tf.nn.conv2d(x, k, strides=[1, 3, 3, 1],
                        padding='SAME')

output_image = conv(image, kernel)
output_pool = tf.nn.avg_pool(image, ksize=[1, 3, 3, 1],
                             strides=[1, 3, 3, 1],
                             padding='SAME')

kernel_data = np.zeros(shape=(3, 3, 3, 3)).astype(np.float32)
kernel_data[:, :, 0, 0] = 1 / 9.
kernel_data[:, :, 1, 1] = 1 / 9.
kernel_data[:, :, 2, 2] = 1 / 9.

with tf.Session() as sess:
    feed_dict = {image: [sample_image], kernel: kernel_data}
    conv_img, pool_img = sess.run([output_image, output_pool],
                                  feed_dict=feed_dict)
    print(conv_img.shape, pool_img.shape)
    plt.subplot(1, 2, 1)
    show(conv_img[0])
    plt.title("conv")
    plt.subplot(1, 2, 2)
    show(pool_img[0])
    plt.title("avg_pool")

# Note that the numerical computation/approximation might
# be slightly different in the two cases