import torch
from model import TestNet

def evaluate_model(model, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in test_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy}%')

if __name__ == "__main__":
    input_dim = 2
    output_dim = 2
    batch_size = 32

    model = TestNet(input_dim, output_dim)
    model.load_state_dict(torch.load('TestNet.pth'))

    test_data = torch.randn(200, input_dim)
    test_labels = (test_data.sum(dim=1) > 0).long()

    test_dataset = torch.utils.data.TensorDataset(test_data, test_labels)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    evaluate_model(model, test_loader)
