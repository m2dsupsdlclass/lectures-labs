class Siamese(tf.keras.Model):
    def __init__(self, shared_conv):
        super().__init__(self, name="siamese")
        self.conv = shared_conv
        self.dot = Dot(axes=-1, normalize=True)

    def call(self, inputs):
        i1, i2 = inputs
        x1 = self.conv(i1)
        x2 = self.conv(i2)
        return self.dot([x1, x2])


model = Siamese(shared_conv)
model.compile(loss=contrastive_loss, optimizer='rmsprop', metrics=[accuracy_sim])

###### binary classification instead of cosine
#
# out = Lambda(lambda x: K.abs(x[0] - x[1]))([x1, x2])
# out = Dense(1, activation="sigmoid")(out)
#
# model = Model(inputs=[i1, i2], outputs=out)
# model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
