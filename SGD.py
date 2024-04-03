import torch
from utils import SimpleNN, train_model
from main import train_loader, val_loader
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from utils import criterion, evaluate_model

# SGD on a simple classification problem

class MySGD:
    def __init__(self, params, lr):
        self.params = list(params)
        self.lr = lr

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

    def step(self):
        # 1. Loop over all parameters
        # 2. Update the parameters with SGD
        # Hint: use p.data and p.grad to access the parameter values and the gradient

        for p in self.params:
            if p.grad is None:
                continue

            p.data = p.data - self.lr * p.grad

        # Some sanity checks
        # The type of p is : <class 'torch.nn.parameter.Parameter'>
        # The type of p.grad is : <class 'torch.Tensor'> (representing the gradient of p)
        # The type of p.data is : <class 'torch.Tensor'>  (representing the value of p)


# Train our model with MySGD on MNIST and evaluate its performance
# on the validation set.

# Set hyperparameters
learning_rate = 0.01
num_epochs = 15

# Initialize model and optimizer
model_sgd = SimpleNN().to(device)
optimizer_sgd = MySGD(model_sgd.parameters(), lr=learning_rate) # Here we use our own SGD optimizer

# Train model
train_loss_sgd = []
val_loss_sgd = []
for epoch in range(num_epochs):
    loss = train_model(model_sgd, train_loader, optimizer_sgd, criterion, device)
    train_loss_sgd.append(loss)
    # Evaluate on validation set
    val_loss, val_accuracy = evaluate_model(model_sgd, val_loader, criterion, device)
    val_loss_sgd.append(val_loss)
    print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")


