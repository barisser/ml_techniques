import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# Check if CUDA is available and set PyTorch to use GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_set = torchvision.datasets.MNIST(root='../data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)

test_set = torchvision.datasets.MNIST(root='../data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False)

def to_one_hot(labels, num_classes=10):
    return torch.eye(num_classes)[labels].to(device)

def shorten(x, max, border, sharpness=None):
    if sharpness is None:
        if border < 0:
            return torch.zeros_like(x)
        else:
            limit = int(torch.ceil(border * max))
            x[:, limit + 1:] = 0
            return x
    m = (torch.arange(max) - border * max) * sharpness
    y1 = 1 - torch.sigmoid(m)
    return x * y1

# 2. Define a simple Neural Network with a training function
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.max = 1000
        self.w1 = nn.Parameter(torch.tensor(0.3))
        self.w2 = nn.Parameter(torch.tensor(0.1))
        self.target_sum = torch.tensor(0.2)
        self.fc1 = nn.Linear(28*28, self.max, bias=False)
        self.fc2 = nn.Linear(self.max, self.max, bias=False)
        self.fc3 = nn.Linear(self.max, 10, bias=False)

        self.criterion = nn.CrossEntropyLoss()
        #self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.optimizer = optim.SGD(self.parameters(), lr=0.01)

    def non_zero_weights(self):
        return sum((torch.abs(p) > 0).sum().item() for p in self.parameters())

    # def forward(self, x):
    #     x = x.view(-1, 28*28)
    #     x = torch.relu(self.fc1(x))
    #     x = torch.relu(self.fc2(x))
    #     x = self.fc3(x)
    #     return x


    def forward(self, x, sharpness=None):
        x = x.view(-1, 28*28)
        x1 = torch.relu(self.fc1(x))
        x2 = shorten(x1, self.max, self.w1, sharpness)
        x3 = torch.relu(self.fc2(x2))

        x4 = shorten(x3, self.max, self.w2, sharpness)
        x5 = self.fc3(x4)
    #    import pdb;pdb.set_trace()
        return x5

    def train_model(self, train_loader, num_epochs=3):
        for epoch in range(num_epochs):
            for i, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward pass
                sharpness = 0.01 * epoch ** 1.5
                outputs = self(inputs, sharpness)
                loss = self.criterion(outputs, to_one_hot(labels, 10))
                    
                capacity_overage = torch.clamp(self.w1 + self.w2 - self.target_sum, min=0)
                capacity_penalty = torch.exp(capacity_overage)
                
                loss += capacity_penalty

                # Backward pass and optimize
                loss.backward()
                #import pdb;pdb.set_trace()

                # with torch.no_grad():
                #     for name, param in model.named_parameters():
                #         if "w1" == name or "w2" == name:
                #             param.grad *= 100.0  # Scale down gradients for parameters in "specific_layer"

                self.optimizer.step()
            # non_zero_weights = sum((torch.abs(p) > 1e-5).sum().item() for p in self.parameters())
            # print("nonzero: {}".format(non_zero_weights))

            train_accuracy = self.get_accuracy(train_loader)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Train Accuracy: {train_accuracy:.2f}%")
            print("{}    {}   {}".format(self.w1, self.w2, self.w1 + self.w2))
            #import pdb;pdb.set_trace()
        print("Training completed!")

    def get_accuracy(self, loader):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in loader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return 100 * correct / total

# Create and train the model
model = SimpleNN().to(device)
model.train_model(train_loader, 5)

# Evaluate accuracy on test set
test_accuracy = model.get_accuracy(test_loader)
print(f"Test Accuracy: {test_accuracy:.2f}%")
