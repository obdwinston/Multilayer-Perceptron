# Multilayer Perceptron

Multilayer perceptron to solve classification and regression problems.

- [x] Single/Multioutput Regression
- [x] Binary/Multilabel/Multiclass Classification
- [x] L1/L2 Regularisation
- [ ] Dropout Regularisation
- [x] Mini-Batch Gradient Descent (GD)
- [x] Adam Optimisation (Momentum GD + RMS Propagation)
- [x] Learning Rate Decay
- [ ] Batch Normalisation
- [ ] and more...

![animation](https://github.com/obdwinston/Multilayer-Perceptron/assets/104728656/d9d2cff9-ec53-461d-b136-94a981ca94f3)
_The baseline verification was performed with [Scikit-Learn's MLPClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html) on a generated dataset (3 features, 4 classes). Test accuracy scores for both models were about 0.9, with comparable runtimes._

## Mini-Batch Gradient Descent

![image](https://github.com/obdwinston/Multilayer-Perceptron/assets/104728656/bb37e703-9f05-4672-8529-b942421f624b)
_Faster convergence rates were achieved with increasing batches, but converged at higher costs. Peak accuracy was achieved with 2 batches, with no further incentive to increase batches (and hence runtimes) with decreasing accuracy._

## Adam Optimisation

![image](https://github.com/obdwinston/Multilayer-Perceptron/assets/104728656/f03de763-d9f4-4763-b37d-8943d3bcb10b)
_All Adam cases converged much faster to lower costs, and peak accuracy was achieved with the first Adam case (β₁ = 0.9, β₂ = 0.99). Of note, zero betas do not correspond to vanilla gradient descent, due to a resulting scaling factor acting on the learning rate that is dependent on both epsilon (small value to prevent division by zero in Adam) and the gradient itself:_ $W=W-\alpha(\frac{1}{|dW| + \epsilon})dW$.

## L1/L2 Regularisation

![regularisation](https://github.com/obdwinston/Multilayer-Perceptron/assets/104728656/d7e53c83-72e4-44bb-804f-0c36c0fd2602)
_Decreasing regularisation increased converged cost. Peak accuracy was achieved for Case 2, typically attributed to a balance between overfitment (Case 1) and underfitment (Cases 3 and 4)._

## Learning Rate Decay

![image](https://github.com/obdwinston/Multilayer-Perceptron/assets/104728656/2e5764f1-a424-42d3-9b6d-26a3d011bbcb)
_Exponential decay resulted in the slowest convergence rate due to the rapid decrease in learning rate. Scheduled decay had a fast initial convergence rate, while gradually reducing (or relaxing) the learning rate in a step-wise manner._
