# Multilayer Perceptron

Multilayer perceptron to solve classification and regression problems.

- [x] Single/Multioutput Regression
- [x] Binary/Multilabel/Multiclass Classification
- [x] L1/L2 Regularisation
- [ ] Dropout Regularisation
- [x] Mini-Batch Gradient Descent (GD)
- [x] Adam Optimisation (Momentum GD + RMS Propagation)
- [ ] Learning Rate Decay
- [ ] Batch Normalisation
- [ ] and more...

![animation](https://github.com/obdwinston/Multilayer-Perceptron/assets/104728656/d9d2cff9-ec53-461d-b136-94a981ca94f3)
_15 Jun 24: The baseline verification was performed with [Scikit-Learn's MLPClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html) on a generated dataset (3 features, 4 classes). Test accuracy scores for both models were about 0.9, with comparable runtimes._

## Mini-Batch Gradient Descent
![image](https://github.com/obdwinston/Multilayer-Perceptron/assets/104728656/bb37e703-9f05-4672-8529-b942421f624b)
_17 Jun 24: Mini-batch gradient descent was added. Faster convergence rates were achieved with increasing batches, but converged at higher costs. Peak accuracy was achieved with 2 batches, with no further incentive to increase batches (and hence runtimes) with decreasing accuracy._

## Adam Optimisation
![image](https://github.com/obdwinston/Multilayer-Perceptron/assets/104728656/8006d38b-291c-45c3-a208-a9240f7b5605)
_18 Jun 24: Adam optimisation was added. All cases with Adam (non-zero betas) achieved much faster convergence rates, and converged at lower costs. Higher accuracies were achieved for vanilla gradient descent (zero betas) and the first Adam case. Interestingly, setting zero betas will not exactly resolve to vanilla gradient descent as one would expect, due to a resultant scaling factor acting on the learning rate that is dependent on both epsilon (small value introduced to prevent division by zero in Adam) and the gradient itself. This will result in slightly different cost and accuracy, but likely faster convergence rate, than vanilla gradient descent per se. A separate if statement was implemented to check for zero betas before directly executing vanilla gradient descent, which explains the otherwise higher runtime._

![image](https://github.com/obdwinston/Multilayer-Perceptron/assets/104728656/ff510b83-b871-405d-9313-f5566d9430b6)
