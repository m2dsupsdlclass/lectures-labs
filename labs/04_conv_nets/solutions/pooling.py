max_pool = MaxPool2D(2, strides=2, input_shape=(None, None, 3))
img_in = np.expand_dims(sample_image, 0)
img_out = max_pool(img_in).numpy()

print("input shape:", img_in.shape)
print("output shape:", img_out.shape)

fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(10, 5))
ax0.imshow(img_in[0].astype('uint8'))
ax1.imshow(img_out[0].astype('uint8'));

# it is not possible to build a max pooling with a regular convolution
# however it is possible to build average pooling with well  
# chosen strides and kernel
