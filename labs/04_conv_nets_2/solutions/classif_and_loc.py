# test acc: 0.898, mean iou: 0.457, acc_valid: 0.496
# This is by no means the best model; however the lack
# of input data forbids us to build much deeper networks

def classif_and_loc(num_classes):
    model_input = Input(shape=(7,7,2048))
    x = GlobalAveragePooling2D()(model_input)
    
    x = Dropout(0.2)(x)
    head_classes = Dense(num_classes, activation="softmax", name="head_classes")(x)
    
    y = Convolution2D(4, 1, 1, activation='relu', name='hidden_conv')(model_input)
    y = Flatten()(y)
    y = Dropout(0.2)(y)
    head_boxes = Dense(4, name="head_boxes")(y)
    
    model = Model(model_input, output = [head_classes, head_boxes], name="resnet_loc")
    model.compile(optimizer="adam", loss=['categorical_crossentropy', "mse"], 
                  loss_weights=[1., 1/(224*224)]) 
    return model

model = classif_and_loc(5)

history = model.fit(x = inputs, y=[out_cls, out_boxes], 
                    validation_data=(test_inputs, [test_cls, test_boxes]), 
                    batch_size=batch_size, nb_epoch=30, verbose=2)

compute_acc(train=True)
compute_acc(train=False)
