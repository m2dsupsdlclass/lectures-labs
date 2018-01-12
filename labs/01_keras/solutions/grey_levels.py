# axis=2 is the third axis as indexing starts at 0 in Python
grey_image = image.mean(axis=2)
print("dtype:", grey_image.dtype)
print("shape:", grey_image.shape)

# Each 64 bit floating point takes 8 bytes in memory. 
# (450 * 800 * 1) * (64 // 8) in bytes
print("size: {:0.3f} MB".format(grey_image.nbytes / 1e6))
print("min value:", grey_image.min())
print("max value:", grey_image.max())

plt.imshow(grey_image, cmap=plt.cm.Greys_r);
