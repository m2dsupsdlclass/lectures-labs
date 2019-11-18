class TripletNetwork(tf.keras.Model):
    def __init__(self, shared_conv):
        super().__init__(self, name="tripletnetwork")
        
        self.shared_conv = shared_conv
        self.dot = Dot(axes=-1, normalize=True)
        self.cosine_triple_loss = Lambda(cosine_triplet_loss, output_shape=(1,))
        
    def call(self, inputs):
        anchor, positive, negative = inputs
        
        anchor = self.shared_conv(anchor)
        positive = self.shared_conv(positive)
        negative = self.shared_conv(negative)

        
        pos_sim = self.dot([anchor, positive])
        neg_sim = self.dot([anchor, negative])

        return self.cosine_triple_loss([pos_sim, neg_sim])
   
model_triplet = TripletNetwork(shared_conv)
model_triplet.compile(loss=identity_loss, optimizer="rmsprop")