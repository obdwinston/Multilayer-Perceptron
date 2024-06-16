# Multilayer Perceptron

Multilayer perceptron to solve classification and regression problems.

- [x] Single/Multioutput Regression
- [x] Logistic/Multilabel/Multiclass Classification
- [x] L1/L2 Regularisation
- [ ] Dropout Regularisation
- [ ] Mini-Batch Gradient Descent (GD)
- [ ] Adam Optimisation (Momentum GD + RMS Propagation)
- [ ] Learning Rate Decay
- [ ] Batch Normalisation
- [ ] and more...

The aim of this coding project was to put together in practice what I have learnt from several machine learning courses. It is always beneficial (and enjoyable) to take a moment to observe theory in action, and to validate the results against reliable sources. I learned to appreciate the importance of vectorising array operations, which is crucial for efficient computing and scalability. Vectorisation also enhances code conciseness and readability by eliminating the need for nested for loops. The following verification was performed with [Scikit-Learn's MLPClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html) on a generated dataset (3 features, 4 classes). Test accuracy scores for both models were about 0.9. Runtimes were comparable for the current dataset.
