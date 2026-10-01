import json
import torch

# Input features and target label
x1 = torch.tensor(1.0, dtype=torch.float64)
x2 = torch.tensor(0.0, dtype=torch.float64)
y = torch.tensor(1.0, dtype=torch.float64)

# Network parameters with gradient tracking
w_h1_x1 = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
w_h1_x2 = torch.tensor(-0.4, dtype=torch.float64, requires_grad=True)
b_h1 = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)

w_h2_x1 = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
w_h2_x2 = torch.tensor(0.8, dtype=torch.float64, requires_grad=True)
b_h2 = torch.tensor(-0.2, dtype=torch.float64, requires_grad=True)

w_o_h1 = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
w_o_h2 = torch.tensor(-0.6, dtype=torch.float64, requires_grad=True)
b_o = torch.tensor(0.05, dtype=torch.float64, requires_grad=True)

# Explicit forward pass
z_h1 = w_h1_x1 * x1 + w_h1_x2 * x2 + b_h1
a_h1 = torch.sigmoid(z_h1)

z_h2 = w_h2_x1 * x1 + w_h2_x2 * x2 + b_h2
a_h2 = torch.sigmoid(z_h2)

z_o = w_o_h1 * a_h1 + w_o_h2 * a_h2 + b_o
y_hat = torch.sigmoid(z_o)

# Binary cross entropy loss
loss = torch.nn.functional.binary_cross_entropy(y_hat, y)

# Backpropagation
loss.backward()

# Reference dictionary
reference_data = {
    "input": [float(x1.item()), float(x2.item())],
    "label": float(y.item()),
    "forward": {
        "z_h1": float(z_h1.item()),
        "a_h1": float(a_h1.item()),
        "z_h2": float(z_h2.item()),
        "a_h2": float(a_h2.item()),
        "z_o": float(z_o.item()),
        "y_hat": float(y_hat.item()),
    },
    "loss": float(loss.item()),
    "gradients": {
        "w_h1_x1": float(w_h1_x1.grad.item()),
        "w_h1_x2": float(w_h1_x2.grad.item()),
        "b_h1": float(b_h1.grad.item()),
        "w_h2_x1": float(w_h2_x1.grad.item()),
        "w_h2_x2": float(w_h2_x2.grad.item()),
        "b_h2": float(b_h2.grad.item()),
        "w_o_h1": float(w_o_h1.grad.item()),
        "w_o_h2": float(w_o_h2.grad.item()),
        "b_o": float(b_o.grad.item()),
    },
}

# Output only valid JSON to stdout
print(json.dumps(reference_data, indent=2))
