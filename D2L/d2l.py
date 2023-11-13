from d2l import torch as d2l
import torch
import matplotlib.pyplot as plt
# import mlp_toolkits.axes_grid1 as ImageGrid

import torch
import torchvision
import torchvision.transforms as transforms
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
        root= "./D2L/images/image_data_folder",
        transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
    )
    tiny_image_dataloader = torch.utils.data.DataLoader(tiny_image_dataset, batch_size=6, shuffle=False)
    images, labels = next(iter(tiny_image_dataloader))
    # img_show(torchvision.utils.make_grid(images, nrow=2))
    print(tiny_image_dataset.classes)

class SillyDataset(torch.utils.data.Dataset):
    def __init__(self, shape=(224, 224)):
        super(SillyDataset, self).__init__()
        self.shape = shape

    def __getitem__(self, index):
        # Return a random tensor of requested shape
        return torch.rand(*self.shape)

    def __len__(self):
        # This is how the dataloader knows to stop vending at the end of an epoch
        return 1000

def explore_d2l(args):
    if args.dataset is not None:
        if args.dataset == "fmnist":
            fmnist()
        elif args.dataset == "imgs":
            images()
        elif args.dataset == "silly":
            silly_dataset = SillyDataset(shape=(10,10))
            silly_dataloader = torch.utils.data.DataLoader(silly_dataset, batch_size=32)
            
            for data in silly_dataloader:
                print(data.shape)
                print(data)
                break
        else:
            print(f"Unrecognized dataset: {args.dataset}")
        return

    pass