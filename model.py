import sys
import numpy as np

np.random.seed(42)

class Perceptron:
    
    def __init__(self, configuration):

        self.type = configuration.get('type', 'classifier')
        self.units = configuration.get('units', [10])
        self.activations = configuration.get('activations', ['relu'])
        self.alpha = configuration.get('alpha', 1e-2)
        self.lambda_1 = configuration.get('lambda_1', 1e-3)
        self.lambda_2 = configuration.get('lambda_2', 1e-3)
        self.batches = configuration.get('batches', 1)
        self.iterations = configuration.get('iterations', int(1e3))
        self.verbose = configuration.get('verbose', True)

        self.epsilon = 1e-10

    def initialise_model(self, X, y):
        
        self.data = (X, y)
        self.nx, mx = X.shape
        self.ny, my = y.shape
        assert mx == my, 'Number of examples of X and y must be equal'
        self.nc = int(np.max(np.sum(y, axis=0, keepdims=True)))
        
        self.units = [self.nx] + self.units + [self.ny]

        self.subtype = ''
        if self.type == 'classifier':
            if (self.ny == 1) or (self.ny > 1 and self.nc > 1):
                self.subtype = 'multilabel' # logistic/multilabel
                self.activations = ['linear'] + self.activations + ['sigmoid']
            elif (self.ny > 1 and self.nc == 1):
                self.subtype = 'multiclass' # multiclass
                self.activations = ['linear'] + self.activations + ['softmax']
        else: # regressor
            self.activations = ['linear'] + self.activations + ['linear']

        self.parameters = {}
        self.gradients = {}
        self.caches = []
        self.costs = []

        L = len(self.activations)
        for l in range(1, L):
            self.parameters['W' + str(l)] = np.random.randn(self.units[l], self.units[l - 1]) / np.sqrt(self.units[l - 1]) # He initialisation
            self.parameters['b' + str(l)] = np.zeros((self.units[l], 1))
    
    def create_batches(self):

        X = self.data[0]
        y = self.data[1]
        _, m = X.shape
        
        permutation = list(np.random.permutation(m))
        X_shuffled = X[:, permutation]
        y_shuffled = y[:, permutation]

        batch_size = m // self.batches
        assert batch_size >= 1, 'Batch size must be greater than or equal to 1'

        mini_batches = []
        for i in range(self.batches):
            start = i * batch_size
            end = start + batch_size

            X_batch = X_shuffled[:, start:end]
            y_batch = y_shuffled[:, start:end]
            
            mini_batches.append((X_batch, y_batch))
        
        return mini_batches

    def forward_propagation(self):

        self.caches = [] # reset cache

        def activate(Z, activation):
            if activation == 'relu': return np.maximum(0, Z)
            elif activation == 'tanh': return np.tanh(Z)
            elif activation == 'sigmoid': return 1 / (1 + np.exp(-Z))
            elif activation == 'softmax':
                E = np.exp(Z - np.max(Z, axis=0, keepdims=True))
                return E / np.sum(E, axis=0, keepdims=True)
            else: return Z # linear
                
        A_prev = self.X
        L = len(self.activations)
        for l in range(1, L):
            W = self.parameters['W' + str(l)]
            b = self.parameters['b' + str(l)]
            Z = np.dot(W, A_prev) + b

            cache = (A_prev, W, b, Z)
            self.caches.append(cache)

            activation = self.activations[l]
            A = activate(Z, activation)

            A_prev = A
        
        self.AL = A

    def compute_cost(self):

        # non-regularisation term
        if self.subtype == 'multilabel':
            cost = -(1 / self.m) * np.sum(self.y * np.log(self.AL + self.epsilon) + (1 - self.y) * np.log(1 - self.AL + self.epsilon)) # logistic loss
        elif self.subtype == 'multiclass':
            cost = -(1 / self.m) * np.sum(self.y * np.log(self.AL + self.epsilon)) # softmax loss
        else: # regressor
            cost = (1 / self.m) * np.sum((self.AL - self.y) ** 2) # mse loss

        # regularisation term
        regularisation = 0
        L = len(self.activations)

        # L2 regularisation
        if (self.lambda_2 > 0):
            for l in range(1, L):
                W = self.parameters['W' + str(l)]
                regularisation += (self.lambda_2 / (2 * self.m)) * np.sum(np.square(W))
        
        # L1 regularisation
        if (self.lambda_1 > 0):
            for l in range(1, L):
                W = self.parameters['W' + str(l)]
                regularisation += (self.lambda_1 / self.m) * np.sum(np.abs(W))

        cost += regularisation
        cost = float(np.squeeze(cost))
        if np.isnan(cost):
            sys.exit()
        
        return cost

    def backward_propagation(self):

        def derivative(A, Z, activation):
            if activation == 'relu': return np.where(A > 0, 1, 0)
            elif activation == 'tanh': return (1 - A ** 2)
            elif activation == 'sigmoid': return A * (1 - A)
            elif activation == 'softmax':
                E = np.exp(Z - np.max(Z, axis=0, keepdims=True))
                S = E / np.sum(E, axis=0, keepdims=True)
                return S - self.y # dAdZ = dZ since dA set to 1 for softmax
            else: return np.ones_like(A) # linear
        
        A = self.AL
        if self.subtype == 'multilabel':
            dA = -(self.y / (self.AL + self.epsilon) - (1 - self.y) / (1 - self.AL + self.epsilon)) # logistic loss derivative
        elif self.subtype == 'multiclass':
            dA = 1 # dZ computed without dA for softmax
        else: # regressor
            dA = self.AL - self.y # mse loss derivative
        
        L = len(self.activations)
        for l in reversed(range(1, L)):
            activation = self.activations[l]
            A_prev, W, b, Z = self.caches[l - 1]

            dAdZ = derivative(A, Z, activation)
            dZ = dA * dAdZ # dA set to 1 for softmax

            dW = (1 / self.m) * np.dot(dZ, A_prev.T) + (self.lambda_2 / self.m) * W + (self.lambda_1 / self.m) * np.sign(W)
            db = (1 / self.m) * np.sum(dZ, axis=1, keepdims=True)

            self.gradients['dW' + str(l)] = dW
            self.gradients['db' + str(l)] = db

            A = A_prev
            dA = np.dot(W.T, dZ) # W^[l] = dZ^[l] / dA^[l - 1]
        
    def update_parameters(self):
        
        L = len(self.activations)
        for l in range(1, L):
            self.parameters['W' + str(l)] -= self.alpha * self.gradients['dW' + str(l)]
            self.parameters['b' + str(l)] -= self.alpha * self.gradients['db' + str(l)]
    
    def train_model(self, X, y):

        self.initialise_model(X, y)

        mini_batches = self.create_batches()

        for i in range(self.iterations):
            cost = 0
            for mini_batch in mini_batches:
                self.X, self.y = mini_batch
                _, self.m = self.X.shape

                self.forward_propagation()
                self.backward_propagation()
                self.update_parameters()

                cost += self.compute_cost()

            if (i % 1000 == 0):
                self.costs.append(cost / self.batches)
                print(f'Cost at Iteration {i}: {self.costs[-1]: .10f}')

    def predict_model(self, X):

        self.X = X
        self.forward_propagation()

        if self.subtype == 'multilabel':
            return (self.AL > 0.5).astype(int)
        elif self.subtype == 'multiclass':
            return np.argmax(self.AL, axis=0)
        else: # regressor
            return self.AL
