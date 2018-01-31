image = tf.placeholder(tf.float32, [None, None, None, 3])
kernel = tf.placeholder(tf.float32, [3, 3, 3])

def conv(x, k):
    k = tf.reshape(k, shape=[3, 3, 3, 1])
    return tf.nn.depthwise_conv2d(x, k, strides=[1,2,2,1],
                                  padding='SAME')

def conv_valid(x, k):
    k = tf.reshape(k, shape=[3, 3, 3, 1])
    return tf.nn.depthwise_conv2d(x, k, strides=[1,2,2,1],
                                  padding='VALID')

output_image = conv(image, kernel)
output_image_valid = conv_valid(image, kernel)
kernel_data = np.zeros(shape=(3, 3, 3)).astype(np.float32)

# identity kernel: ones only in the center of the filter
kernel_data[1, 1, :] = 1
print('Identity 3x3x3 kernel:')
print(np.transpose(kernel_data, (2, 0, 1)))

with tf.Session() as sess:
    feed_dict = {image: [sample_image], kernel: kernel_data}
    conv_img, conv_img_valid = sess.run([output_image, output_image_valid],
                                        feed_dict=feed_dict)

    print("Shape of result with SAME padding:", conv_img.shape)
    print("Shape of result with VALID padding:", conv_img_valid.shape)
    show(conv_img[0])

# We observe that the stride divided the size of the image by 2
# In the case of 'VALID' padding mode, no padding is added, so 
# the size of the ouput image is actually 1 less because of the
# kernel size