"""Loading saved brains, including ones from before the look-ahead inputs existed."""
import numpy as np
import torch
import config

BASE_INPUTS = len(config.RAY_ANGLES) + 2        # rays + speed + heading: the original 9


def saved_input_size(path):
    """How many inputs the saved network expects, read from its first layer's weights."""
    if path.endswith(".npz"):
        return np.load(path)["arr_0"].shape[0]                    # NumPy W is (inputs, outputs)
    return torch.load(path, weights_only=True)["layers.0.weight"].shape[1]   # PyTorch: (outputs, inputs)


def lookahead_for(n_inputs):
    """Which look-ahead gates a network with this many inputs was trained with."""
    extra = n_inputs - BASE_INPUTS
    if extra == 0:
        return []                                                  # an older model
    if extra == len(config.LOOKAHEAD_GATES):
        return list(config.LOOKAHEAD_GATES)
    raise ValueError(f"A network with {n_inputs} inputs doesn't match LOOKAHEAD_GATES in config.py")


def load_brain(agent, path):
    """Load saved weights into the agent. If the saved brain has FEWER inputs than the agent
    (an old model, from before look-ahead), widen it: the new inputs get weight 0, so it starts
    out driving exactly as before and learns to use them during training."""
    old_inputs = saved_input_size(path)
    if agent.backend == "numpy":
        data = np.load(path)
        for i, p in enumerate(agent.online.parameters()):
            saved = data[f"arr_{i}"]
            if i == 0 and saved.shape[0] < p.shape[0]:
                p[...] = 0.0
                p[:saved.shape[0]] = saved                         # old input rows, new rows stay 0
            else:
                p[...] = saved
    else:
        state = torch.load(path, weights_only=True)
        weight = state["layers.0.weight"]
        needed = agent.online.layers[0].in_features
        if weight.shape[1] < needed:
            wider = torch.zeros(weight.shape[0], needed)
            wider[:, :weight.shape[1]] = weight                    # old input columns, new ones 0
            state["layers.0.weight"] = wider
        agent.online.load_state_dict(state)
    agent.update_target()
    return old_inputs