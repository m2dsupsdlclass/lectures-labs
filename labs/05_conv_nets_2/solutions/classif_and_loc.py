# test acc: 0.898, mean iou: 0.457, acc_valid: 0.496
# This is by no means the best model; however the lack
# of input data forbids us to build much deeper networks

def classif_and_loc(num_classes):
    model_input = Input(shape=(7, 7, 2048))
    # For single object classification, the exact spatial information
    # held in the feature map is not very important. It's fine to perform
    # global average pooling of the features at the beginning of the
    # classification tower of the model.
    classification_tower = GlobalAveragePooling2D()(model_input)
    
    classification_tower = Dropout(0.2)(classification_tower)
    head_classes = Dense(num_classes, activation="softmax",
                         name="head_classes")(classification_tower)
    
    # We do not use global average pooling in the localization tower of the
    # model so as to preserve coarse grained 7 x 7 spatial information
    # from the last convolutional feature maps computed by the ResNet50
    # model.
    localization_tower = Convolution2D(4, (1, 1), activation='relu',
                                       name='hidden_conv')(model_input)
    localization_tower = Flatten()(localization_tower)
    localization_tower = Dropout(0.2)(localization_tower)
    head_boxes = Dense(4, name="head_boxes")(localization_tower)
    
    model = Model(model_input, outputs = [head_classes, head_boxes], name="resnet_loc")
    model.compile(optimizer="adam", loss=['categorical_crossentropy', "mse"], 
                  loss_weights=[1., 1 / (224 * 224)]) 
    return model


better_model = classif_and_loc(5)
history = better_model.fit(x = inputs, y=[out_cls, out_boxes], 
                           validation_data=(test_inputs, [test_cls, test_boxes]), 
                           batch_size=batch_size, epochs=30, verbose=2)

compute_acc(better_model, train=True)
compute_acc(better_model, train=False)
