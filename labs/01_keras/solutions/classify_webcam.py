def classify():
    image = camera_grab(camera_id=0, fallback_filename='laptop.jpeg')
    image_224 = resize(image, (224, 224), preserve_range=True, mode='reflect')
    image_224_batch = np.expand_dims(image_224, axis=0)

    preprocessed_batch = preprocess_input(image_224_batch)
    preds = model.predict(preprocessed_batch)
    print(decode_predictions(preds, top=5))

classify()
