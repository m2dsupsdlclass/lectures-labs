# Proportion of zeros in the vector of the first image:
print("proportion of zeros (first image):", np.mean(out_tensors[0] == 0.0))

# Proportion of negative values in the full representation tensor
print("proportion of strictly negative values:", np.mean(out_tensors < 0.0))

# For all representations:
plt.hist(np.mean(out_tensors == 0.0, axis=1))
plt.title("Fraction of zero values per image vector");

# These 0 values come from the different ReLU units.
# They propagate through the layers, and there can be many.
# If a network has too many of them, a lot of computation
# / memory is wasted.