# My-Journey-In-Machine-Learing
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

# Clean data (no noise)
X_clean, y_clean = make_regression(n_samples=100, n_features=1, noise=0, random_state=42)

# Noisy data
X_noisy, y_noisy = make_regression(n_samples=100, n_features=1, noise=20, random_state=42)

# Plot both
plt.figure(figsize=(12, 5))

# Plot clean data
plt.subplot(1, 2, 1)
plt.scatter(X_clean, y_clean, color='blue', label='Clean Data')
plt.title("Linear Data WITHOUT Noise")
plt.xlabel("Feature X")
plt.ylabel("Target y")
plt.legend()

# Plot noisy data
plt.subplot(1, 2, 2)
plt.scatter(X_noisy, y_noisy, color='red', label='Noisy Data')
plt.title("Linear Data WITH Noise")
plt.xlabel("Feature X")
plt.ylabel("Target y")
plt.legend()

plt.tight_layout()
plt.show()
