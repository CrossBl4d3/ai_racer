import os
import numpy as np
import torch
import torch.nn as nn
import config
from env.racing_env import RacingEnv
from agent.network import QNetwork
from agent.numpy_network import NumpyQNetwork, huber_loss, Adam

N_IN = len(config.RAY_ANGLES) + 2
N_OUT = 5
results = []


def check(name, passed, detail=""):
    results.append(passed)
    print(f"[{'PASS' if passed else 'FAIL'}] {name}  {detail}")


# 1. Shapes and parameter count ------------------------------------------
np_net = NumpyQNetwork(N_IN, N_OUT, seed=0)
torch_net = QNetwork(N_IN, N_OUT)
batch = np.random.rand(64, N_IN).astype(np.float32)

np_out = np_net(batch)
torch_out = torch_net(torch.from_numpy(batch))
check("1. Shapes", np_out.shape == (64, N_OUT) and tuple(torch_out.shape) == (64, N_OUT),
      f"numpy {np_out.shape}, torch {tuple(torch_out.shape)}")
check("1. Parameter count", np_net.num_parameters() == torch_net.num_parameters(),
      f"numpy {np_net.num_parameters():,}, torch {torch_net.num_parameters():,}")

# 2. Real observation -> action --------------------------------------------
env = RacingEnv()
observation = env.reset()
a_np = np_net.act(observation)
a_torch = torch_net.act(observation)
check("2. Observation -> action", 0 <= a_np < N_OUT and 0 <= a_torch < N_OUT,
      f"numpy picks {a_np}, torch picks {a_torch}")

# 3. Gradient check (numpy, 64-bit) ----------------------------------------
net64 = NumpyQNetwork(N_IN, N_OUT, seed=1, dtype=np.float64)
rng = np.random.default_rng(2)
x = rng.random((8, N_IN))
y = rng.standard_normal((8, N_OUT))

loss, grad = huber_loss(net64(x), y)
net64.backward(grad)
analytic = [g.copy() for g in net64.gradients()]

eps = 1e-6
worst = 0.0
for p, g in zip(net64.parameters(), analytic):
    for _ in range(15):                                   # 15 random entries per parameter array
        idx = tuple(rng.integers(0, s) for s in p.shape)
        original = p[idx]
        p[idx] = original + eps
        loss_plus, _ = huber_loss(net64(x), y)
        p[idx] = original - eps
        loss_minus, _ = huber_loss(net64(x), y)
        p[idx] = original
        numeric = (loss_plus - loss_minus) / (2 * eps)
        relative = abs(numeric - g[idx]) / max(1e-8, abs(numeric) + abs(g[idx]))
        worst = max(worst, relative)
check("3. Gradient check", worst < 1e-5, f"worst relative error {worst:.2e}")

# 4. Equivalence: numpy vs torch (outputs AND gradients) -------------------
np_net.load_from_torch(torch_net)
x = np.random.rand(16, N_IN).astype(np.float32)
y = np.random.randn(16, N_OUT).astype(np.float32)

np_pred = np_net(x)
torch_pred = torch_net(torch.from_numpy(x))
output_diff = np.abs(np_pred - torch_pred.detach().numpy()).max()
check("4. Same outputs", output_diff < 1e-5, f"max difference {output_diff:.2e}")

_, grad = huber_loss(np_pred, y)
np_net.backward(grad)
torch_net.zero_grad()
nn.HuberLoss()(torch_pred, torch.from_numpy(y)).backward()
first_np = np_net.layers[0].dW
first_torch = torch_net.linear_layers()[0].weight.grad.numpy().T
grad_diff = np.abs(first_np - first_torch).max()
check("4. Same gradients", grad_diff < 1e-5, f"max difference {grad_diff:.2e}")

# 5. Overfit test ------------------------------------------------------------
X = np.random.rand(10, N_IN).astype(np.float32)
Y = (np.random.randn(10, N_OUT) * 2).astype(np.float32)

net = NumpyQNetwork(N_IN, N_OUT, seed=3)
optimizer = Adam(net.parameters(), lr=1e-3)
start_loss, _ = huber_loss(net(X), Y)
for _ in range(1000):
    loss, grad = huber_loss(net(X), Y)
    net.backward(grad)
    optimizer.step(net.gradients())
check("5. Overfit (numpy + your Adam)", loss < 0.01 * start_loss, f"loss {start_loss:.4f} -> {loss:.6f}")

tnet = QNetwork(N_IN, N_OUT)
toptimizer = torch.optim.Adam(tnet.parameters(), lr=1e-3)
tX, tY = torch.from_numpy(X), torch.from_numpy(Y)
tstart = nn.HuberLoss()(tnet(tX), tY).item()
for _ in range(1000):
    tloss = nn.HuberLoss()(tnet(tX), tY)
    toptimizer.zero_grad()
    tloss.backward()
    toptimizer.step()
check("5. Overfit (torch)", tloss.item() < 0.01 * tstart, f"loss {tstart:.4f} -> {tloss.item():.6f}")

# 6. Copy test -------------------------------------------------------------
a, b = NumpyQNetwork(N_IN, N_OUT, seed=10), NumpyQNetwork(N_IN, N_OUT, seed=11)
b.copy_from(a)
ta, tb = QNetwork(N_IN, N_OUT), QNetwork(N_IN, N_OUT)
tb.copy_from(ta)
same_np = np.array_equal(a(batch), b(batch))
same_torch = torch.equal(ta(torch.from_numpy(batch)), tb(torch.from_numpy(batch)))
check("6. Copy", same_np and same_torch, f"numpy {same_np}, torch {same_torch}")

# 7. Save / load -----------------------------------------------------------
os.makedirs("models", exist_ok=True)
a.save("models/test_numpy.npz")
fresh = NumpyQNetwork(N_IN, N_OUT, seed=99)
fresh.load("models/test_numpy.npz")
ta.save("models/test_torch.pt")
tfresh = QNetwork(N_IN, N_OUT)
tfresh.load("models/test_torch.pt")
same_np = np.array_equal(a(batch), fresh(batch))
same_torch = torch.equal(ta(torch.from_numpy(batch)), tfresh(torch.from_numpy(batch)))
os.remove("models/test_numpy.npz")
os.remove("models/test_torch.pt")
check("7. Save / load", same_np and same_torch, f"numpy {same_np}, torch {same_torch}")

print(f"\n{sum(results)} / {len(results)} checks passed")