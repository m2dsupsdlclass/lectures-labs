def get_adaptable_network(input_shape=x_source_train.shape[1:]):
    inputs = Input(shape=input_shape)
    x = Conv2D(32, 5, padding='same', activation='relu', name='conv2d_1')(inputs)
    x = MaxPool2D(pool_size=2, strides=2, name='max_pooling2d_1')(x)
    x = Conv2D(48, 5, padding='same', activation='relu', name='conv2d_2')(x)
    x = MaxPool2D(pool_size=2, strides=2, name='max_pooling2d_2')(x)
    features = Flatten(name='flatten_1')(x)
    x = Dense(100, activation='relu', name='dense_digits_1')(features)
    x = Dense(100, activation='relu', name='dense_digits_2')(x)
    digits_classifier = Dense(10, activation="softmax", name="digits_classifier")(x)

    domain_branch = Dense(100, activation="relu", name="dense_domain")(GradReverse()(features))
    domain_classifier = Dense(1, activation="sigmoid", name="domain_classifier")(domain_branch)

    return Model(inputs=inputs, outputs=[digits_classifier, domain_classifier])

model = get_adaptable_network()
model.summary()