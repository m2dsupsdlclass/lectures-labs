anchor = Input((60, 60, 3), name='anchor')
positive = Input((60, 60, 3), name='positive')
negative = Input((60, 60, 3), name='negative')

a = shared_conv2(anchor)
p = shared_conv2(positive)
n = shared_conv2(negative)

pos_sim = Dot(axes=-1, normalize=True)([a,p])
neg_sim = Dot(axes=-1, normalize=True)([a,n])

loss = Lambda(cosine_triplet_loss,
              output_shape=(1,))(
             [pos_sim,neg_sim])

model_triplet = Model(
    inputs=[anchor, positive, negative],
    outputs=loss)

model_triplet.compile(loss=identity_loss, optimizer="rmsprop")
