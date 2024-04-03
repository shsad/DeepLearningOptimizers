import torch
from utils import SimpleNN, train_model
from main import train_loader, val_loader
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from utils import criterion, evaluate_model

# Adam class

class MyAdam:
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.params = list(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}  # Placeholder for the first moment vector
        self.v = {}  # Placeholder for the second moment vector
        self.t = 0  # Initialize time step

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

    def step(self):
        self.t += 1  # Increment timestep
        for p in self.params:
            if p.grad is None:
                continue

            # Initialize moment vectors if encountering this parameter for the first time
            if p not in self.m:
                self.m[p] = torch.zeros_like(p.data)
            if p not in self.v:
                self.v[p] = torch.zeros_like(p.data)

            # NOTE
            # You can access the gradient of the parameter by p.grad.data
            # To update or access the first and second moment vectors, use self.m[p] and self.v[p] respectively

            # Update biased first and second moment estimates
            self.m[p] = self.beta1 * self.m[p] + (1 - self.beta1) * p.grad.data
            self.v[p] = self.beta2 * self.v[p] + (1 - self.beta2) * p.grad.data ** 2

            # Compute bias-corrected first and second moment estimates
            m_hat = self.m[p] / (1 - self.beta1 ** self.t)
            v_hat = self.v[p] / (1 - self.beta2 ** self.t)

            # Update the parameter
            p.data -= self.lr * m_hat / (torch.sqrt(v_hat) + self.epsilon)


# Set hyperparameters
learning_rate = 0.001
num_epochs = 15
# Initialize model and optimizer
model_adam = SimpleNN().to(device)
# Here we use our own Adam optimizer
optimizer_adam = MyAdam(model_adam.parameters(), lr=learning_rate)
# Train model
train_loss_adam = []
val_loss_adam = []
for epoch in range(num_epochs):
    loss = train_model(model_adam, train_loader, optimizer_adam, criterion, device)
    train_loss_adam.append(loss)

    val_loss, val_accuracy = evaluate_model(model_adam, val_loader, criterion, device)
    val_loss_adam.append(val_loss)
    print(
        f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")