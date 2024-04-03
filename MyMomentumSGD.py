import torch
from utils import SimpleNN, train_model
from main import train_loader, val_loader
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from utils import criterion, evaluate_model

# MyMomentumSGD class

class MyMomentumSGD:
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
            self.velocity[i] = self.momentum * self.velocity[i] + self.lr * param.grad
            # 2. Update the parameters
            param.data -= self.velocity[i]


# Train our model with our new optimizer and compare results with MySGD

# Set hyperparameters
learning_rate = 0.01
num_epochs = 15

# Initialize model and optimizer
model_msgd = SimpleNN().to(device)
# Here we use our own SGD with Momentum optimizer
optimizer_msgd = MyMomentumSGD(model_msgd.parameters(), learning_rate=learning_rate)

# Train model
train_loss_msgd = []
val_loss_msgd = []
for epoch in range(num_epochs):
    loss = train_model(model_msgd, train_loader, optimizer_msgd, criterion, device)
    train_loss_msgd.append(loss)

    val_loss, val_accuracy = evaluate_model(model_msgd, val_loader, criterion, device)
    val_loss_msgd.append(val_loss)
    print(
        f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")