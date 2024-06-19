import numpy as np
import matplotlib.pyplot as plt
import time

from model import Perceptron
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# data generation

X, y = make_classification(
    n_samples=200,
    n_classes=4,
    n_clusters_per_class=2,
    n_features=3,
    n_informative=3,
    n_redundant=0,
    n_repeated=0,
    class_sep=1.2,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X1, y1 = X_train.T, np.eye(int(np.max(y_train)) + 1)[y_train.flatten()].T
X2, y2 = X_test.T, np.eye(int(np.max(y_test)) + 1)[y_test.flatten()].T
print('X shape:', X1.shape, X2.shape)
print('y shape:', y1.shape, y2.shape)

# current model

fig, axs = plt.subplots(1, 2, figsize=(12, 6))

axs[0].set_title('Cost vs. Iteration')
axs[0].set_xlabel('Iteration')
axs[0].set_ylabel('Cost')
axs[0].grid(True)

axs[1].set_title('Accuracy and Runtime vs. Keep Probability')
axs[1].set_xlabel('Keep Probability')
axs[1].grid(True)

accuracies = []
runtimes = []

lambdas_list = []
values = np.arange(0, 1, 0.01)

for value in values:
    lambdas_list += [(value, value)]

keep_prob_list = [1.0, 0.95, 0.85, 0.75]

for i, keep_prob in enumerate(keep_prob_list):
    start_time = time.time()

    configuration = {
        'type': 'classifier',
        'units': [20, 13, 7, 5],
        'activations': ['tanh'] * 4,
        'alpha': 1e-2,
        'lambda_1': 3e-1,
        'lambda_2': 3e-1,
        'keep_prob': keep_prob, 
        'beta_1': 0.9,
        'beta_2': 0.99,
        'batches': 1,
        'iterations': int(1e4),
        'verbose': True
    }

    model = Perceptron(configuration)
    model.train_model(X1, y1)
    y_pred = model.predict_model(X2)

    y_pred = np.eye(int(np.max(y_pred + 1)))[y_pred.flatten()].T
    accuracy = 1 - (np.sum(np.abs(y2 - y_pred)) // 2) / y2.shape[1]
    accuracies.append(accuracy)

    end_time = time.time()
    runtime = end_time - start_time
    runtimes.append(runtime)

    iterations = np.arange(0, len(model.costs) * 10, 10)
    label = f'p = {keep_prob}'
    axs[0].plot(iterations, model.costs, label=label)

axs[0].legend()

axs[1].plot(range(len(keep_prob_list)), runtimes, label='Runtime', marker='o')
axs[1].set_ylabel('Runtime (seconds)', color='tab:blue')
axs[1].tick_params(axis='y', labelcolor='tab:blue')

axs2 = axs[1].twinx()
axs2.plot(range(len(keep_prob_list)), accuracies, color='tab:red', label='Accuracy', marker='o')
axs2.set_xticks(range(len(keep_prob_list)))
axs2.set_xticklabels(['1.0', '0.95', '0.85', '0.75'])
axs2.set_ylabel('Accuracy', color='tab:red')
axs2.tick_params(axis='y', labelcolor='tab:red')

axs[1].legend(loc='upper right')
axs2.legend(loc='lower right')
print(accuracies)
plt.tight_layout()
plt.show()
