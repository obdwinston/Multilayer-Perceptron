import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.animation import FuncAnimation, PillowWriter

from model import Perceptron
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

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

configuration = {
    'type': 'classifier',
    'units': [20, 13, 7, 5],
    'activations': ['tanh'] * 4,
    'alpha': 1e-2,
    'lambda_1': 3e-1,
    'lambda_2': 3e-1,
    'batches': 1,
    'iterations': int(1e5),
    'verbose': True
}

model = Perceptron(configuration)
model.train_model(X1, y1)
y_pred = model.predict_model(X2)

accuracy = []
y_pred = np.eye(int(np.max(y_pred + 1)))[y_pred.flatten()].T
accuracy.append(1 - (np.sum(np.abs(y2 - y_pred)) // 2) / y2.shape[1])
print(f'Model accuracy: {accuracy[0]: .3f}')

# scikit model

print('Running Scikit model...')

mlp = MLPClassifier(
    hidden_layer_sizes=(20, 13, 7, 5),
    activation='tanh',
    max_iter=int(1e6),
    n_iter_no_change=int(1e5),
    tol=1.,
    alpha=0.3, # regularisation
    random_state=42,
    verbose=False)
mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_test)

accuracy.append(accuracy_score(y_test, y_pred))
print(f'Scikit accuracy: {accuracy[1]: .3f}')

# decision boundary

print('Plotting decision boundary...')

fig = plt.figure(figsize=(11, 7), constrained_layout=True)
fig.suptitle('Decision Boundary', fontsize=16, fontweight='bold')
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')

xx, yy = np.meshgrid(
    np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 50),
    np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 50))
z_min, z_max = X[:, 2].min(), X[:, 2].max()
colormap = ListedColormap(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])

def plot_decision_boundary(x, ax, fixed_feature):
    ax.clear()
    for class_label in range(4):
        X_class = X[y == class_label]
        ax.scatter(X_class[:, 0], X_class[:, 1], X_class[:, 2], label=f'Class {class_label + 1}', alpha=0.8, color=colormap(class_label))
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Feature 3')
    ax.legend()
    ax.legend(loc='upper right', bbox_to_anchor=(1, 1))

    zz = np.full(xx.shape, fixed_feature)
    grid = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]

    if x == 1:
        predictions = model.predict_model(grid.T)
        ax.set_title('Current Model', fontweight='bold')
        ax.text2D(0.5, 0.95, f'Accuracy: {accuracy[0]: .3f}', transform=ax.transAxes, ha='center')
    else:
        predictions = mlp.predict(grid)
        ax.set_title('Scikit Model', fontweight='bold')
        ax.text2D(0.5, 0.95, f'Accuracy: {accuracy[1]: .3f}', transform=ax.transAxes, ha='center')
    predictions = predictions.reshape(xx.shape)

    facecolors = colormap(predictions)
    ax.plot_surface(xx, yy, zz, facecolors=facecolors, alpha=0.3, rstride=1, cstride=1, linewidth=0, antialiased=False)

fixed_features = np.linspace(z_min, z_max, 50)

def update(frame):
    fixed_feature = fixed_features[frame]
    plot_decision_boundary(1, ax1, fixed_feature)
    plot_decision_boundary(2, ax2, fixed_feature)
    return ax1, ax2

ani = FuncAnimation(fig, update, frames=len(fixed_features), blit=False, repeat=True)
ani.save('./saves/animation.gif', writer=PillowWriter(fps=5))
plt.show()
