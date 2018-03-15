inp = Input((60, 60, 3), dtype='float32')
x = Conv2D(16, 3, activation="relu", padding="same")(inp)
x = Conv2D(16, 3, activation="relu", padding="same")(x)
x = MaxPool2D((2,2))(x) # 30,30
x = Conv2D(32, 3, activation="relu", padding="same")(x)
x = Conv2D(32, 3, activation="relu", padding="same")(x)
x = MaxPool2D((2,2))(x) # 15,15
x = Conv2D(64, 3, activation="relu", padding="same")(x)
x = Conv2D(64, 3, activation="relu", padding="same")(x)
x = MaxPool2D((2,2))(x) # 8,8
x = Conv2D(64, 3, activation="relu", padding="same")(x)
x = Conv2D(32, 3, activation="relu", padding="same")(x)
x = Flatten()(x)
x = Dropout(0.2)(x)
x = Dense(50)(x)
shared_conv = Model(inputs=inp, outputs=x)
