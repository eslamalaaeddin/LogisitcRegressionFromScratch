import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

## Generating Random Data
class_1 = np.random.random(20) * 2 + 1  # 20 samples for first class
class_2 = np.random.random(20) * 2 - 0.5

data = pd.DataFrame()
data['x'] = np.concatenate([class_1, class_2])  # 40 samples
data['y'] = [1] * 20 + [0] * 20  # 20: Pos, 20: Neg

print(data.head())
# Plotting data
plt.scatter(data.x, data.y, s=5)
plt.show()


# P =  e^(B0 + B1 * X) / (1 + e^(B0 + B1 * X))
def calculate_gradient_log_likelihood(curr_betas, data):
    numerator = np.exp(curr_betas[0] + curr_betas[1] * data.x)
    p = numerator / (1 + numerator)

    partial_0 = np.sum(data.y - p)  # Partial Derivative of Log Likelihood With Respect To B0
    partial_1 = np.sum((data.y - p) * data.x)  # Partial Derivative of Log Likelihood With Respect To B1

    return np.array([partial_0, partial_1])


# Initial Values Of B0 and B1
betas = np.array([0.0, 0.0])
diff = np.inf  # Represents Gradients: If Grads Are Too Low: Stop Optimizing B0, B1
eta = 0.1

while diff > .001:
    grad = calculate_gradient_log_likelihood(betas, data)
    diff = abs(grad).sum()
    betas += eta * grad

print(betas)

plt.scatter(data.x, data.y, s=5)

x_vals = np.arange(data.x.min(), data.x.max(), .01)
p_vals = 1 / (1 + np.exp(-(betas[0] + betas[1] * x_vals)))
plt.plot(x_vals, p_vals)
plt.show()
clf = LogisticRegression(penalty=None)
clf.fit(np.array(data.x).reshape(-1, 1), data.y)
print('beta_0: %s' % clf.intercept_[0])
print('beta_1: %s' % clf.coef_[0][0])

plt.scatter(data.x, data.y, s=5)

x_vals = np.arange(data.x.min(), data.x.max(), .01)
p_vals = 1 / (1 + np.exp(-(clf.intercept_[0] + clf.coef_[0][0] * x_vals)))
plt.plot(x_vals, p_vals)
plt.show()
