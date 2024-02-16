import argparse
import sys

import torch
from d2l import torch as d2l
import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip, RandomVerticalFlip, ColorJitter, RandomResizedCrop
from torch.utils.data import TensorDataset

def img_show(img):
    plt.rcParams["figure.figsize"] = (15,30)
    out = img.permute(1, 2, 0).numpy()
    plt.imshow(out); plt.show()

def fmnist():
    fmnist_train = torchvision.datasets.FashionMNIST(
        root='./data', download=True, train=True, transform=transforms.ToTensor())
    fmnist_test = torchvision.datasets.FashionMNIST(
        root='./data', download=True, train=False, transform=transforms.ToTensor())

    # Get a slice of the full dataset to test TensorDataset
    data_tensor = fmnist_train.data[:50]
    target_tensor = fmnist_train.targets[:50]
    tensor_dataset = TensorDataset(data_tensor, target_tensor)
    dataloader = torch.utils.data.DataLoader(
        tensor_dataset,
        batch_size=32,
        shuffle=True,
        drop_last=False)# drop_last=True will drop the last batch if it is not full

    # for data, labels in dataloader:
    #     print(data.shape, labels.shape)

    fmnist_dataloader = torch.utils.data.DataLoader(fmnist_train, batch_size=25, shuffle=True)
    images, labels = next(iter(fmnist_dataloader))
    img_show(torchvision.utils.make_grid(images, nrow=8))


def images():
    tiny_image_dataset = torchvision.datasets.ImageFolder(
        root= "./data/d2l-images/image_data_folder",
        transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
    )
    tiny_image_dataloader = torch.utils.data.DataLoader(tiny_image_dataset, batch_size=6, shuffle=False)
    images, labels = next(iter(tiny_image_dataloader))
    # img_show(torchvision.utils.make_grid(images, nrow=2))
    print(tiny_image_dataset.classes)

def plot_transformed_cifar(transformation):
    # Function to convert tensors into an appropriate numpy array to plot
    def imshow(img):
        out = img.permute(1, 2, 0).numpy()
        plt.imshow(out); plt.show()

    train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transformation)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=False, num_workers=2, drop_last=True)

    # Plot the first batch from the dataloader
    fig = plt.figure(figsize=(12., 12.))
    data_iter = iter(train_loader)
    images, _ = next(data_iter)
    imshow(torchvision.utils.make_grid(images, nrow=8))

def cifar10():
    transformations = Compose([
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        RandomResizedCrop((32,32), scale=(0.5, 1), ratio=(0.5,2)),
        ToTensor()
    ])
    plot_transformed_cifar(transformations)

def explore_d2l(args):
    if args.dataset is not None:
        if args.dataset == "fmnist":
            fmnist()
        elif args.dataset == "imgs":
            images()
        elif args.dataset == "cifar10":
            cifar10()
        else:
            print(f"Unrecognized dataset: {args.dataset}")
        return

def main(args):
    parser = argparse.ArgumentParser(description="Dive Into Deep Learning")
    commands = parser.add_subparsers(dest="cmd")

    explore_cmd = commands.add_parser("explore", help="Explore d2l data")
    explore_cmd.add_argument("--dataset", "-ds", required=False)
    explore_cmd.set_defaults(action=explore_d2l)
    
    args = parser.parse_args()
    if not hasattr(args, "action"):
        parser.print_help()
        return 1

    args.action(args)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
