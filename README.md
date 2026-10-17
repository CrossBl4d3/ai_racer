# AI Racer 🏎️

A top-down racing game built with Pygame, and an AI driver trained with
**reinforcement learning (Deep Q-Learning)** to race it, using a neural
network **written from scratch in NumPy** (forward pass, backpropagation
and the Adam optimizer, all by hand), alongside an equivalent PyTorch version.

## 🏆 Results

**Champion: my from-scratch NumPy network. Best lap 4.95 s vs the human record of 5.68 s.**

Tournament: 50 races per model, each from a random start position on the track.

| Model          | Network          | Finished | Avg 3-lap time | Best lap | Wins |
|----------------|------------------|:--------:|:--------------:|:--------:|:----:|
| **numpy2**     | NumPy (scratch)  | **50/50**| **15.33 s**    | **4.95 s** | **42** |
| numpy3         | NumPy (scratch)  | 47/50    | 15.40 s        | 4.95 s   | 6    |
| run2           | PyTorch          | 50/50    | 15.46 s        | 5.02 s   | 3    |
| run1           | PyTorch          | 28/50    | 15.45 s        | 5.00 s   | 0    |
| numpy1         | NumPy (scratch)  | 48/50    | 15.57 s        | 5.00 s   | 0    |
| *Human (me)*   | *keyboard*       |          |                | *5.68 s* |      |

**Key findings**
- The AI beat my best lap within ~12