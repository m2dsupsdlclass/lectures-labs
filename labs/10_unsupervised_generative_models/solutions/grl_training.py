from tensorflow.keras.losses import SparseCategoricalCrossentropy, BinaryCrossentropy
from tensorflow.keras.metrics import Mean, Accuracy


optimizer = SGD(lr=0.01, momentum=0.9, nesterov=True)

cce = SparseCategoricalCrossentropy()
bce = BinaryCrossentropy()

model.compile(
    optimizer=optimizer,
    loss=[cce, bce],
    metrics=["accuracy", "accuracy"]
)

def train_epoch(source_train_generator, target_train_generator):
    global lambda_factor, global_step

    # Keras provide helpful classes to monitor various metrics:
    epoch_source_digits = tf.keras.metrics.Mean()
    epoch_source_domains = tf.keras.metrics.Mean()
    epoch_target_domains = tf.keras.metrics.Mean()
    epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    # Fetch all trainable variables but those used uniquely for the digits classification:
    variables_but_classifier = list(filter(lambda x: "digits" not in x.name, model.trainable_variables))

    loss_record = collections.defaultdict(list)

    for i, data in enumerate(zip(source_train_generator, target_train_generator)):
        source_data, target_data = data
        # Training digits classifier & domain classifier on source:
        x_source, y_source, d_source = source_data

        with tf.GradientTape() as tape:
            digits_prob, domains_probs = model(x_source)
            digits_loss = cce(y_source, digits_prob)
            domains_loss = bce(d_source, domains_probs)
            source_loss = digits_loss + 0.2 * domains_loss

        gradients = tape.gradient(source_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        epoch_source_digits(digits_loss)
        epoch_source_domains(domains_loss)
        epoch_accuracy(y_source, digits_prob)

        # Training domain classifier on target:
        x_target, d_target = target_data
        with tf.GradientTape() as tape:
            _, domains_probs = model(x_target)
            target_loss = 0.2 * bce(d_target, domains_probs)

        gradients = tape.gradient(target_loss, variables_but_classifier)
        optimizer.apply_gradients(zip(gradients, variables_but_classifier))

        epoch_target_domains(target_loss)

    print("Source digits loss={}, Source Accuracy={}, Source domain loss={}, Target domain loss={}".format(
        epoch_source_digits.result(), epoch_accuracy.result(),
        epoch_source_domains.result(), epoch_target_domains.result()))


for epoch in range(epochs):
    print("Epoch: {}".format(epoch), end=" ")
    loss_record = train_epoch(source_train_generator, target_train_generator)
