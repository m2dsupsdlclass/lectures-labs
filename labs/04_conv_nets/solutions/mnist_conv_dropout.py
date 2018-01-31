# Model definition
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# Loss function and optimizer
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_true)
loss = tf.reduce_mean(cross_entropy)

train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

# Metrics
correct_prediction = tf.equal(tf.argmax(y_conv, 1),
                              tf.argmax(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Main training loop
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(500):
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            # no dropout
            feed_dict = {x:batch[0], y_true: batch[1], keep_prob: 1.0}
            train_accuracy = accuracy.eval(feed_dict=feed_dict)
            print("update %d, training accuracy %g"
                  % (i, train_accuracy))
        # dropout
        feed_dict = {x:batch[0], y_true: batch[1], keep_prob: 0.5}
        train_step.run(feed_dict = feed_dict)
    # no dropout
    feed_dict = {x: mnist.test.images,
                 y_true: mnist.test.labels, keep_prob: 1.0}
    print("test accuracy %g" % accuracy.eval(feed_dict = feed_dict))
