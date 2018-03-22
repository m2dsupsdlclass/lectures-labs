for i in [0, 4, 6]:
    plt.figure(figsize=(10, 2))
    for j in range(5):
        plt.subplot(1, 5, j + 1)
        plt.imshow(x_train[y_train == i][j], cmap="gray")
        plt.axis("off")
    plt.suptitle("Samples from class %d (%s):"
                 % (i, id_to_labels[i]))
    plt.show()