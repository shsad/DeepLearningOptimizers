import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Simple FFNN baseline model.
# Single hidden layer with 128 units and ReLU activation.
# Output layer with 10 units (one for each class) and
# softmax activation.
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create an instance of the model and move it to the device
model = SimpleNN().to(device)
print(model)

# Loop over the parameters
for name, param in model.named_parameters():
    print(f"Name: {name}, Shape: {param.shape}")

for param in model.parameters():
    print(param.shape)


# Helper functions: Training and Evaluation Routines
# Training and evaluation functions, used later for training and
# evaluating the model. We will use the cross-entropy loss.

# Define the loss and metrics
criterion = nn.CrossEntropyLoss()

# Define the training function
def train_model(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    return running_loss / len(dataloader.dataset)

# Define the evaluation function
def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
    accuracy = correct_predictions / len(dataloader.dataset)
    return running_loss / len(dataloader.dataset), accuracy

