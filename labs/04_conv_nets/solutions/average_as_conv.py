avg_pool = AvgPool2D(3, strides=3, input_shape=(None, None, 3))

img_in = np.expand_dims(sample_image, 0)
img_out_avg_pool = avg_pool(img_in).numpy()

# Same operation implemented with a convolution

def my_init(shape=(3, 3, 3, 3), dtype=None):
    array = np.zeros(shape=shape, dtype="float32")
    array[:, :, 0, 0] = 1 / 9.
    array[:, :, 1, 1] = 1 / 9.
    array[:, :, 2, 2] = 1 / 9.
    return array

# padding="valid" means no padding.
# In our case we don't need padding:
# See formula: w' = (w - k + 2 * p) / s + 1
# With k=3, s=3, and p=0, the output volume w' is
# w' = w / 3
conv_avg = Conv2D(kernel_size=3, filters=3, strides=3,
           padding="valid", kernel_initializer=my_init,
           input_shape=(None, None, 3))

img_out_conv = conv_avg(np.expand_dims(sample_image, 0)).numpy()

print("input shape:", img_in.shape)
print("output avg pool shape:", img_out_avg_pool.shape)
print("output conv shape:", img_out_conv.shape)

fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, figsize=(10, 5))
ax0.imshow(img_in[0].astype('uint8'))
ax1.imshow(img_out_avg_pool[0].astype('uint8'))
ax2.imshow(img_out_conv[0].astype('uint8'));

# Note that the numerical computation/approximation might
# be slightly different in the two cases
print("Avg pool is similar to Conv ? -", np.allclose(img_out_avg_pool, img_out_conv))
