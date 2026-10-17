"""A Q-network written from scratch with NumPy: forward pass, backpropagation and Adam."""
import numpy as np
import config


class Linear:
    """Fully connected layer: Y = X·W + b"""

    def __init__(self, n_in, n_out, rng, dtype):
        # He initialisation: keeps signals a healthy size through ReLU layers
        self.W = (rng.standard_normal((n_in, n_out)) * np.sqrt(2.0 / n_in)).astype(dtype)
        self.b = np.zeros(n_out, dtype=dtype)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.x = None

    def forward(self, x):
        self.x = x                           # remembered for the backward pass
        return x @ self.W + self.b

    def backward(self, grad_out):
        self.dW = self.x.T @ grad_out        # dL/dW = Xᵀ · dL/dY
        self.db = grad_out.sum(axis=0)       # dL/db = column sums of dL/dY
        return grad_out @ self.W.T           # dL/dX = dL/dY · Wᵀ  (passed to the layer below)

    def params(self):
        return [self.W, self.b]

    def grads(self):
        return [self.dW, self.db]


class ReLU:
    """ReLU activation: max(0, x)"""

    def forward(self, x):
        self.mask = x > 0                    # remember where the input was positive
        return x * self.mask

    def backward(self, grad_out):
        return grad_out * self.mask          # gradient flows only where input was > 0

    def params(self):
        return []

    def grads(self):
        return []


class NumpyQNetwork:
    def __init__(self, n_inputs, n_actions, hidden_sizes=None, seed=None, dtype=np.float32):
        if hidden_sizes is None:
            hidden_sizes = config.HIDDEN_SIZES
        rng = np.random.default_rng(seed)
        self.dtype = dtype

        sizes = [n_inputs] + list(hidden_sizes) + [n_actions]
        self.layers = []
        for i in range(len(sizes) - 1):
            self.layers.append(Linear(sizes[i], sizes[i + 1], rng, dtype))
            if i < len(sizes) - 2:           # ReLU after every layer except the last
                self.layers.append(ReLU())

    def forward(self, x):
        x = np.asarray(x, dtype=self.dtype)
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def __call__(self, x):
        return self.forward(x)

    def backward(self, grad):
        for layer in reversed(self.layers):  # the chain rule, from the output backwards
            grad = layer.backward(grad)

    def parameters(self):
        return [p for layer in self.layers for p in layer.params()]

    def gradients(self):
        return [g for layer in self.layers for g in layer.grads()]

    def num_parameters(self):
        return sum(p.size for p in self.parameters())

    def act(self, observation):
        q_values = self.forward(np.asarray(observation)[None, :])   # add a batch dimension
        return int(np.argmax(q_values[0]))

    def copy_from(self, other):
        for mine, theirs in zip(self.parameters(), other.parameters()):
            mine[...] = theirs               # copy values in place

    def save(self, path):
        np.savez(path, *self.parameters())

    def load(self, path):
        data = np.load(path)
        for i, p in enumerate(self.parameters()):
            p[...] = data[f"arr_{i}"]

    def load_from_torch(self, torch_network):
        """Copy weights from the PyTorch QNetwork (its weights are stored transposed)."""
        my_linears = [layer for layer in self.layers if isinstance(layer, Linear)]
        for mine, theirs in zip(my_linears, torch_network.linear_layers()):
            mine.W[...] = theirs.weight.detach().numpy().T
            mine.b[...] = theirs.bias.detach().numpy()


def huber_loss(prediction, target, delta=1.0):
    """Returns (loss, dLoss/dPrediction). Like MSE for small errors, gradient capped at ±delta."""
    error = prediction - target
    abs_error = np.abs(error)
    quadratic = np.minimum(abs_error, delta)
    linear = abs_error - quadratic
    loss = np.mean(0.5 * quadratic ** 2 + delta * linear)
    grad = np.clip(error, -delta, delta) / error.size
    return loss, grad


class Adam:
    """The Adam optimizer, written by hand."""

    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = [np.zeros_like(p) for p in params]   # running average of gradients
        self.v = [np.zeros_like(p) for p in params]   # running average of squared gradients
        self.t = 0

    def step(self, grads):
        self.t += 1
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            m *= self.beta1
            m += (1 - self.beta1) * g
            v *= self.beta2
            v += (1 - self.beta2) * g * g
            m_hat = m / (1 - self.beta1 ** self.t)    # bias correction for the early steps
            v_hat = v / (1 - self.beta2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)