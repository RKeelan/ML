import argparse
import glob
import logging
from pathlib import Path
import sys

from d2l import torch as d2l
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from tqdm import tqdm

CHECKPOINT_DIR = Path('./checkpoints/')

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(in_features=16*5*5, out_features=120),
            nn.Sigmoid(),
            nn.Linear(in_features=120, out_features=84),
            nn.Sigmoid(),
            nn.Linear(in_features=84, out_features=10)
        )

    def forward(self, x):
        return self.net(x)


def get_dataset():
    train_dataset = torchvision.datasets.MNIST("data", train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = torchvision.datasets.MNIST("data", train=False, transform=transforms.ToTensor(), download=True)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_dataloader, test_dataloader


def img_show(img):
    plt.rcParams["figure.figsize"] = (15,30)
    out = img.permute(1, 2, 0).numpy()
    plt.imshow(out); plt.show()


def explore_lenet(args):
    data_loader, _ = get_dataset()
    images, _ = next(iter(data_loader))
    print(images.shape)
    img_show(torchvision.utils.make_grid(images, nrow=8))


def evaluate(net, data_iter, device=None):
    if isinstance(net, torch.nn.Module):
        net.eval()
        if not device:
            device = next(iter(net.parameters())).device
    
    metric = d2l.Accumulator(2)
    for X, y in data_iter:
        if isinstance(X, list):
            X = [x.to(device) for x in X]
        else:
            X = X.to(device)
        y = y.to(device)
        metric.add(d2l.accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


def train(net, train_iter, test_iter, num_epochs, lr, device):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)
    print("Training on", device)
    net.to(device)
    
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    # This doesn't seem to work when running from the command line (but I think it'd work in a jupyter notebook)
    # animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
    #                         legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = d2l.Timer(), len(train_iter)
    for epoch in range(num_epochs):
        # Sum of training loss, sum of training accuracy, no. of examples
        metric = d2l.Accumulator(3)
        net.train()
        with tqdm(total=num_batches, desc=f"Epoch {epoch+1}/{num_epochs}", unit=" img") as pbar:
            for i, (X, y) in enumerate(train_iter):
                timer.start()
                optimizer.zero_grad()
                X, y = X.to(device), y.to(device).long()
                y_hat = net(X)
                l = loss(y_hat, y)
                l.backward()
                optimizer.step()

                pbar.update(i)

                with torch.no_grad():
                    metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
                timer.stop()
                train_l = metric[0] / metric[2]
                train_acc = metric[1] / metric[2]
                pbar.set_postfix(**{'loss (batch)': train_l})
                # if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                #     animator.add(epoch + (i + 1) / num_batches,
                #                 (train_l, train_acc, None))
            test_acc = evaluate(net, test_iter)
            # animator.add(epoch + 1, (None, None, test_acc))
            Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
            state_dict = net.state_dict()
            torch.save(state_dict, f"{CHECKPOINT_DIR}/lenet_checkpoint_{epoch}.pth")

    print(f"loss {train_l:.3f}, train acc {train_acc:.3f}, "
          f"test acc {test_acc:.3f}")
    print(f"{metric[2] * num_epochs / timer.sum():.1f} examples/sec "
          f"on {str(device)}")
    
    Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
    state_dict = net.state_dict()
    torch.save(state_dict, str(CHECKPOINT_DIR / "lenet_checkpoint_final.pth"))
    logging.info(f'Checkpoint {epoch} saved!')


def train_lenet(device, args):
    logging.info("Training LeNet")
    lr = 0.005
    num_epochs = args.epochs
    data_loader, test_loader = get_dataset()
    train(LeNet(), data_loader, test_loader, num_epochs, lr, device)


def lenet_predict(device, args):
    input_files = glob.glob(f"{args.input}\*.jpg")
    model_path = f"{CHECKPOINT_DIR}/lenet_checkpoint_final.pth"
    state_dict = torch.load(model_path)
    model = LeNet()
    model.load_state_dict(state_dict)
    logging.info(f"Model loaded from {model_path}")

    model.eval()
    transform = transforms.Compose([transforms.ToTensor()])
    for filename in input_files:
        img = Image.open(filename)
        img = img.convert("L")
        tensor_img = transform(img)
        tensor_img = tensor_img.unsqueeze(0)
        tensor_img = tensor_img.to(device)

        with torch.no_grad():
            output = model(tensor_img).cpu()
            predicted = torch.argmax(output)
            print(f"{filename}: {predicted}")

def main(args):
    parser = argparse.ArgumentParser(description="LeNet")
    models = parser.add_subparsers(dest="model_type")

    train_parser = models.add_parser("train", help="Train a LeNet")
    train_parser.add_argument("--epochs", "-e", metavar="INPUT", help="Number of epochs", type=int, default=30)
    train_parser.set_defaults(action=train_lenet)

    predict_parser = models.add_parser("predict", help="Predict with LeNet")
    predict_parser.add_argument("--input", "-i", metavar="INPUT", help="Input directory", required=True)
    predict_parser.set_defaults(action=lenet_predict)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f'Using device {device}')

    if not hasattr(args, "action"):
        parser.print_help()
        return 1
    
    args.action(device, args)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))