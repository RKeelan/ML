import torch
import torch.nn as nn
import torch.optim as optim
from model import TestNet
from models import TestNet

def train_model(model: nn.Module, model_name: str, train_loader, optimizer, criterion, epochs: int):
    for epoch in range(epochs):
        for i, (data, labels) in enumerate(train_loader):
            outputs = model(data)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i+1) % 10 == 0:
                print(f'{model_name} Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    # Save the model
    torch.save(model.state_dict(), f"{model_name}.pth")

if __name__ == "__main__":
    input_dim = 2
    output_dim = 2
    batch_size = 32
    learning_rate = 0.001
    epochs = 10

    train_data = torch.randn(1000, input_dim)
    train_labels = (train_data.sum(dim=1) > 0).long()

    train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    model = TestNet(input_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_model(model, "TestNet", train_loader, optimizer, criterion, epochs)
