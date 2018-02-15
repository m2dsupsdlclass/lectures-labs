def plot_iou(boxA, boxB, img_size=(10, 10)):
    iou_value = iou(boxA, boxB)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title("IoU: {:0.3f}".format(iou_value))
    ax.set_ylim(0, 10)
    ax.set_xlim(0, 10)
    wA = boxA[2] - boxA[0]
    hA = boxA[3] - boxA[1]
    ax.add_patch(plt.Rectangle((boxA[0], boxA[1]), wA, hA,
                               color='blue', alpha=0.5))
    wB = boxB[2] - boxB[0]
    hB = boxB[3] - boxB[1]
    ax.add_patch(plt.Rectangle((boxB[0], boxB[1]), wB, hB,
                               color='red', alpha=0.5))


plot_iou([2, 2, 8, 8], [3, 3, 7, 9])
plot_iou([2, 2, 8, 8], [2, 2, 8, 9])
plot_iou([2, 2, 8, 8], [0, 1, 1, 5])
plot_iou([2, 2, 8, 8], [1, 1, 10, 3])