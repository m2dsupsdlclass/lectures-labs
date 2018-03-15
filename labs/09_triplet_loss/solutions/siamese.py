i1 = Input((60, 60, 3), dtype='float32')
i2 = Input((60, 60, 3), dtype='float32')

x1 = shared_conv(i1)
x2 = shared_conv(i2)

out = Dot(axes=-1, normalize=True)([x1, x2])

model = Model(inputs=[i1, i2], outputs=out)
model.compile(loss=contrastive_loss, optimizer='rmsprop', metrics=[accuracy_sim])

###### binary classification instead of cosine
#
# out = Lambda(lambda x: K.abs(x[0] - x[1]))([x1, x2])
# out = Dense(1, activation="sigmoid")(out)
#
# model = Model(inputs=[i1, i2], outputs=out)
# model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])