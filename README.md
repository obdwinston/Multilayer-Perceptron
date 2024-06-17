# Multilayer Perceptron

Multilayer perceptron to solve classification and regression problems.

- [x] Single/Multioutput Regression
- [x] Binary/Multilabel/Multiclass Classification
- [x] L1/L2 Regularisation
- [ ] Dropout Regularisation
- [x] Mini-Batch Gradient Descent (GD)
- [ ] Adam Optimisation (Momentum GD + RMS Propagation)
- [ ] Learning Rate Decay
- [ ] Batch Normalisation
- [ ] and more...

![animation](https://github.com/obdwinston/Multilayer-Perceptron/assets/104728656/d9d2cff9-ec53-461d-b136-94a981ca94f3)
_15 Jun 24: The baseline verification was performed with [Scikit-Learn's MLPClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html) on a generated dataset (3 features, 4 classes). Test accuracy scores for both models were about 0.9, with comparable runtimes._

_17 Jun 24: Mini-batch gradient descent was added. Faster convergence rates were achieved with increasing batches, but converged at higher costs. Peak accuracy was achieved with 2 batches, with no further incentive to increase batches (and hence runtimes) with decreasing accuracy._

![image](https://github.com/obdwinston/Multilayer-Perceptron/assets/104728656/ff510b83-b871-405d-9313-f5566d9430b6)
