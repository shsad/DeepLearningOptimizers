import torch
from utils import SimpleNN, train_model
from dataload import train_loader, val_loader
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from utils import criterion, evaluate_model

# (Simplified) Nesterov Accelerated Gradient (NAG) class

class MyNesterovMomentumSGD:
    def __init__(self, params, learning_rate=0.01, momentum=0.9):
        self.params = list(params)
        self.lr = learning_rate
        self.momentum = momentum
        self.velocity = [torch.zeros_like(p.data) for p in self.params if p is not None]

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()

    def step(self):
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue

            # 1. Compute the velocity update
            self.velocity[i] = self.momentum * self.velocity[i] + param.grad

            # 2. Calculate the "corrected" gradient
            corrected_gradient = param.grad + self.momentum * self.velocity[i]

            # 3. Update the parameters
            param.data.add_(corrected_gradient, alpha=-self.lr)


# Set hyperparameters
learning_rate = 0.01
num_epochs = 15

# Initialize model and optimizer
model_nesterov = SimpleNN().to(device)
optimizer_nesterov = MyNesterovMomentumSGD(model_nesterov.parameters(),
                                           learning_rate=learning_rate)  # Here we use our own Nestrov Accelerated Gradient

# Train model
train_loss_nesterov = []
val_loss_nesterov = []
for epoch in range(num_epochs):
    loss = train_model(model_nesterov, train_loader, optimizer_nesterov, criterion, device)
    train_loss_nesterov.append(loss)

    val_loss, val_accuracy = evaluate_model(model_nesterov, val_loader, criterion, device)
    val_loss_nesterov.append(val_loss)
    print(
        f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")