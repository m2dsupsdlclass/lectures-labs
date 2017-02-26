# Proportion of zeros in a representation
print("proportion of zeros", np.mean(out_tensors[0]==0.0))

# For all representations:
plt.hist(np.mean(out_tensors==0.0, axis=1));

# These 0 values come from the different reLU units.
# They propagate through the layers, and there can be many.
# If a network has too many of them, a lot of computation
# / memory is wasted.
