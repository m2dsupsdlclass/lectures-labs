plt.figure()
plt.imshow(lowres_large_range_image.astype(np.uint8), interpolation='nearest');
plt.figure()
plt.imshow(lowres_large_range_image / 255, interpolation='nearest');