class SharedConv(tf.keras.Model):
    def __init__(self):
        super().__init__(self, name="sharedconv")

        self.conv1 = Conv2D(16, 3, activation="relu", padding="same")
        self.conv2 = Conv2D(16, 3, activation="relu", padding="same")
        self.pool1 = MaxPool2D((2,2)) # 30,30
        self.conv3 = Conv2D(32, 3, activation="relu", padding="same")
        self.conv4 = Conv2D(32, 3, activation="relu", padding="same")
        self.pool2 = MaxPool2D((2,2)) # 15,15
        self.conv5 = Conv2D(64, 3, activation="relu", padding="same")
        self.conv6 = Conv2D(64, 3, activation="relu", padding="same")
        self.pool3 = MaxPool2D((2,2)) # 8,8
        self.conv7 = Conv2D(64, 3, activation="relu", padding="same")
        self.conv8 = Conv2D(32, 3, activation="relu", padding="same")
        self.flatten = Flatten()
        self.dropout = Dropout(0.2)
        self.fc = Dense(50)

    def call(self, inputs):
        x = self.pool1(self.conv2(self.conv1(inputs)))
        x = self.pool2(self.conv4(self.conv3(x)))
        x = self.pool3(self.conv6(self.conv5(x)))
        x = self.flatten(self.conv8(self.conv7(x)))

        return self.fc(self.dropout(x))

shared_conv = SharedConv()
