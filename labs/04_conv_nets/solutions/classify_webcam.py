import time


def classify():
    image = camera_grab(camera_id=0, fallback_filename='laptop.jpeg')

    image = resize(image, (224, 224), preserve_range=True, mode='reflect')
    image_batch = np.expand_dims(image.astype(np.float32), axis=0)
    image_batch = preprocess_input(image_batch)

    tic = time.time()
    results = decode_predictions(model(image_batch).numpy())[0]
    toc = time.time()

    fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(9, 5))
    classnames = [r[1] for r in reversed(results)]
    confidences = [r[2] for r in reversed(results)]
    pos = np.arange(len(classnames))

    ax0.barh(pos, confidences)
    ax0.set_yticks(pos)
    ax0.set_yticklabels(classnames)
    ax0.set_xlim(0, 1)
    ax1.imshow(image / 255)
    fig.suptitle("Prediction time: {:0.3}s".format(toc - tic))


classify()
