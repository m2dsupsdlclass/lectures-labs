grey_image = np.mean(image, axis=-1)
print("Shape: {}".format(grey_image.shape))
print("Type: {}".format(grey_image.dtype))
print("image size: {:0.3} MB".format(grey_image.nbytes / 1e6))
print("Min: {}; Max: {}".format(grey_image.min(), grey_image.max()))

plt.imshow(grey_image, cmap=plt.cm.Greys_r)
