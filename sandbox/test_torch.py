import torch
import torch.nn as nn

# --- 1. Setup check ---
torch.manual_seed(0)  # same "random" numbers every run

print("PyTorch version:", torch.__version__)
print("GPU available:", torch.cuda.is_available())

# --- 2. Network size ---
N_INPUTS = 7      # e.g. 5 sensor rays + speed + angle
N_HIDDEN = 64     # neurons in each hidden layer
N_ACTIONS = 5     # e.g. accelerate, left, right, brake, coast

# --- 3. Define the network ---
class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(N_INPUTS, N_HIDDEN),   # 7  -> 64
            nn.ReLU(),
            nn.Linear(N_HIDDEN, N_HIDDEN),   # 64 -> 64
            nn.ReLU(),
            nn.Linear(N_HIDDEN, N_ACTIONS),  # 64 -> 5 (no ReLU: scores can be negative)
        )

    def forward(self, x):
        return self.layers(x)

# --- 4. Create and inspect ---
net = TinyNet()
print(net)

num_params = sum(p.numel() for p in net.parameters())
print("Total trainable parameters:", num_params)

# --- 5. Forward pass ("thinking") ---
fake_observation = torch.rand(1, N_INPUTS)   # shape (batch=1, 7)
print("\nFake observation:", fake_observation)

with torch.no_grad():
    q_values = net(fake_observation)

print("Output shape:", q_values.shape)
print("Action scores:", q_values)

best_action = q_values.argmax(dim=1).item()
print("Chosen action:", best_action)

# --- 6. Learning check ---
target = torch.tensor([[1.0, -1.0, 2.0, 0.5, 0.0]])  # made-up "correct" scores

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

print("\nTraining on one example:")
for step in range(101):
    prediction = net(fake_observation)        # forward pass
    loss = loss_fn(prediction, target)        # how wrong?

    optimizer.zero_grad()                     # clear old gradients
    loss.backward()                           # backpropagation
    optimizer.step()                          # update weights

    if step % 20 == 0:
        print(f"Step {step:3d} | loss = {loss.item():.6f}")

# --- 7. Check what it learned ---
with torch.no_grad():
    final = net(fake_observation)

print("\nTarget:     ", target)
print("Prediction: ", final)
print("Chosen action now:", final.argmax(dim=1).item(), "(should be 2)")
print("\nPyTorch test passed!")