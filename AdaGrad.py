import torch
from utils import SimpleNN, train_model
from main import train_loader, val_loader
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from utils import criterion, evaluate_model

# AdaGrad class

class MyAdaGrad:
    def __init__(self, params, lr=0.01, epsilon=1e-8):
        self.params = list(params)
        self.lr = lr
        self.eps = epsilon
        self.g_squared = {}  # dictionary to store squared gradients for each parameter

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

    def step(self):
        for param in self.params:
            if param.grad is None:
                continue

            # Initialize squared gradient memory for the parameter if it doesn't exist
            if param not in self.g_squared:
                # NOTE: The key of the dictionary is the parameter itself
                self.g_squared[param] = torch.zeros_like(param.data)

            # 1. Update squared gradient memory
            self.g_squared[param] += param.grad.data ** 2

            # 2. Update parameter values
            param.data -= self.lr * param.grad.data / (torch.sqrt(self.g_squared[param]) + self.eps)


# Set hyperparameters
learning_rate = 0.01
num_epochs = 15
# Initialize model and optimizer
model_adagrad = SimpleNN().to(device)
# optimizer_adam = torch.optim.Adam(model_adam.parameters(), lr=learning_rate)
optimizer_adagrad = MyAdaGrad(model_adagrad.parameters(), lr=learning_rate)
# Train model
train_loss_adagrad = []
val_loss_adagrad = []
for epoch in range(num_epochs):
    loss = train_model(model_adagrad, train_loader, optimizer_adagrad, criterion, device)
    train_loss_adagrad.append(loss)

    val_loss, val_accuracy = evaluate_model(model_adagrad, val_loader, criterion, device)
    val_loss_adagrad.append(val_loss)
    print(
        f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")