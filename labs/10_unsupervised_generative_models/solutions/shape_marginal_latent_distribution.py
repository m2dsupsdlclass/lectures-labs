fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, figsize=(12, 4))

# Sample from the latent variable prior
normal_data = np.random.normal(size=(x_train.shape[0], 2))
ax0.scatter(normal_data[:, 0], normal_data[:, 1], alpha=0.1)
ax0.set_title("Samples from the latent prior $p(z)$")
ax0.set_xlim(-4, 4)
ax0.set_ylim(-4, 4)

# Sample a z_i from the conditional posterior for each x_i in the test set:
z = np.vstack([
    np.random.multivariate_normal(
        x_test_encoded[i], np.diag(np.exp(x_test_encoded_log_var[i] / 2)))
    for i in range(x_test_encoded.shape[0])])
ax1.scatter(z[:, 0], z[:, 1], alpha=0.1)
ax1.set_title("Samples from the latent posterior $q(z|x^i)$")
ax1.set_xlim(-4, 4)
ax1.set_ylim(-4, 4)

# Posterior mean value for each sample x_i from the test set:
ax2.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], alpha=0.1)
ax2.set_title("Test samples encoded in latent space")
ax2.set_xlim(-4, 4)
ax2.set_ylim(-4, 4);

# Analysis:
#
# The VAE KL divergence term of the likelihood lower bound objective function
# is trying to force the encoder to match the posterior distribution with the
# prior of the latent variable. In our case we used:
#               Normal(mean=[0, 0], std=diag([1, 1])
# as the prior distribution which means that 99.7% of the points are expected
# to lie within a radius of 3 around the origin of the 2D latent plan.
#
# Selecting different location and scale parameters for the prior (or even
# a different distribution such as the uniform distribution) would impact the
# shape of the encoded data.