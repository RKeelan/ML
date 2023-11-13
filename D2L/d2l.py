from d2l import torch as d2l
import torch
from torch import nn
from torch.nn import functional as F
import torchvision
import torchvision.transforms.functional as TF

def visualize_tensor(x):
    x = x.squeeze(0).permute(1,2,0)
    d2l.plt.imshow(x.detach().numpy()); d2l.plt.show()

def explore_d2l(args):
    d2l.set_figsize((7,5))
    img = d2l.Image.open("data/d2l/cat1.jpg")
    img = TF.to_tensor(img)
    img = img.reshape([1,3,img.shape[1],img.shape[2]])

    out = nn.MaxPool2d(2)(img)
    print(out.shape)
    # No apparent difference
    # visualize_tensor(out)

    out = nn.MaxPool2d(10)(img)
    print(out.shape)
    # Very clear pixelation
    # visualize_tensor(out)

    out = nn.AvgPool2d(10)(img)
    print(out.shape)
    # Different pixelation
    visualize_tensor(out)