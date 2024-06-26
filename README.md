# Multilayer Perceptron

Standard feedforward neural network with fully-connected layers to solve classification and regression problems.

- [x] Single/Multioutput Regression
- [x] Binary/Multilabel/Multiclass Classification
- [x] L1/L2 Regularisation
- [x] Dropout Regularisation
- [x] Mini-Batch Gradient Descent (GD)
- [x] Adam Optimisation (Momentum GD + RMS Propagation)
- [x] Learning Rate Decay

![animation](https://github.com/obdwinston/Multilayer-Perceptron/assets/104728656/93a4f80a-54dc-4258-9bf5-b6e4d68c7b36)
_The baseline verification was performed with [Scikit-Learn's MLPClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html) on a generated dataset (3 features, 4 classes). Test accuracy scores for both models were about 0.9, with comparable runtimes._

## Adam Optimisation

![image](https://github.com/obdwinston/Multilayer-Perceptron/assets/104728656/66a833a9-816f-4218-811d-47b493f1918d)
_All Adam cases converged much faster to lower costs, and highest accuracy was achieved with the first Adam case (β₁ = 0.9, β₂ = 0.99). The lower accuracy of all Adam cases compared to vanilla gradient descent may be attributed to overfitment, which can be mitigated by early stopping at a much lower iteration. Of note, zero betas do not correspond to vanilla gradient descent, due to a resulting scaling factor acting on the learning rate that is dependent on both epsilon (small value to prevent division by zero in Adam) and the gradient itself:_ $W=W-\alpha(\frac{1}{|dW| + \epsilon})dW$.

## Learning Rate Decay

![image](https://github.com/obdwinston/Multilayer-Perceptron/assets/104728656/2e5764f1-a424-42d3-9b6d-26a3d011bbcb)
_Exponential decay resulted in the slowest convergence rate due to the rapid decrease in learning rate. Scheduled decay had a fast initial convergence rate, while gradually reducing (or relaxing) the learning rate in a step-wise manner._

## L1/L2 Regularisation

![image](https://github.com/obdwinston/Multilayer-Perceptron/assets/104728656/d7e53c83-72e4-44bb-804f-0c36c0fd2602)
_Decreasing regularisation increased converged cost. Peak accuracy was achieved for Case 1 (λ₁ = 0.4, λ₂ = 0.4), typically attributed to a balance between overfitment (Case 0) and underfitment (Cases 2 and 3)._

## Dropout Regularisation

![image](https://github.com/obdwinston/Multilayer-Perceptron/assets/104728656/7f6b148f-90d6-4fe9-bee3-9d76779ba4f4)
_A fair bit of noise was introduced to the cost, which is expected given the random dropouts. The effect of increasing regularisation with higher dropout (lower keep probability) can be inferred from the increasing converged costs. Higher average accuracy was achieved for dropout cases, likely because regularisation tends to reduce overfitment._

## Mini-Batch Gradient Descent

![image](https://github.com/obdwinston/Multilayer-Perceptron/assets/104728656/09fa749d-55fc-4ae1-848a-ffeae7121ac2)
_Faster convergence rates were achieved with increasing batches, but converged at higher costs. Peak accuracy was achieved with 3 batches, with no further incentive to increase batches (and hence runtimes) with decreasing accuracy._
