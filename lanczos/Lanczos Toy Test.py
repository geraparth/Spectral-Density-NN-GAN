from matplotlib import pyplot as plt
import seaborn as sns
from torch import optim

import torch
import torch.nn as nn
import lanczos_algorithm


num_samples = 50
num_features = 16
torch.manual_seed(0)
X = torch.normal(0, 1, size=(num_samples, num_features))
y = torch.normal(0, 1, size=(num_samples, 1))


class Network(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(16, 1, bias=False)

    def forward(self, x):

        x = self.fc1(x)
        return x


def loss_fn(model, inputs):

    X_input, y_input = inputs
    preds = model(X_input)

    return torch.nn.MSELoss(reduction='mean')(y_input, preds)


linear_model = Network()
criterion = torch.nn.MSELoss()
optimizer = optim.SGD(linear_model.parameters(), lr=0.001, momentum=0.9)
trainloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, y), batch_size=5, shuffle=True)

V, T = lanczos_algorithm.approximate_hessian(
        linear_model,
        loss_fn,
        [(X, y)],
        order=num_features)

plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
H = torch.matmul(torch.matmul(V, T).T, V)
plt.title("Hessian as estimated by Lanczos")
sns.heatmap(H)
plt.subplot(1,2,2)
plt.title("$2X^TX$")
sns.heatmap(2 * torch.matmul(X.T, X))
plt.savefig("Hessian_estimate.png")
plt.show()
