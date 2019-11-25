from tensorflow.keras.layers import Add, Layer


class ResidualBlock(Layer):
    def __init__(self, n_filters):
        super().__init__(name="ResidualBlock")

        self.conv1 = Conv2D(n_filters, kernel_size=(3, 3), activation="relu", padding="same")
        self.conv2 = Conv2D(n_filters, kernel_size=(3, 3), padding="same")
        self.add = Add()
        self.last_relu = Activation("relu")

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(inputs)

        y = self.add([x, inputs])
        y = self.last_relu(y)

        return y


class MiniResNet(Model):
    def __init__(self, n_filters):
        super().__init__()

        self.conv = Conv2D(n_filters, kernel_size=(5, 5), padding="same")
        self.block = ResidualBlock(n_filters)
        self.flatten = Flatten()
        self.classifier = Dense(10, activation="softmax")

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.block(x)
        x = self.flatten(x)
        y = self.classifier(x)

        return y


mini_resnet = MiniResNet(32)
mini_resnet.build((None, *input_shape))
mini_resnet.summary()
