from d2l import torch as d2l
import torch
from torch import nn
from torch.nn import functional as F
import torchvision
import torchvision.transforms.functional as TF

def explore_d2l(args):
    d2l.set_figsize((7,5))
    img = d2l.Image.open("data/d2l/cat1.jpg")
    img = TF.to_tensor(img)
    print(f"Image shape: {img.shape}")
    original_data = list(img.flatten())
    # print(f"Image data: {original_data[0:100]}")
    img = img.reshape([1,3,img.shape[1],img.shape[2]])
    reshaped_data = list(img.flatten())
    print(f"Image shape: {img.shape}")
    # print(f"Image data: {reshaped_data[0:100]}")
    print(f"Did reshape re-arrange the data? {'No.' if original_data == reshaped_data else 'Yes.'}")

    # conv2d = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, padding=1, bias=False)
    # 3 inputs / 1 output means that if we want to visualize the image, we should do so in grayscale; but the conv2d
    # operation doesn't actually do anything meaningful
    # d2l.plt.imshow(conv2d(img).reshape(img.shape[2], img.shape[3]).detach().numpy(), cmap='gray'); d2l.plt.show()

    # Let's use nn.Conv2d to apply a 3x3 Laplace filter to the image (this is used for edge detection)
    k = torch.Tensor([[-1,-1,-1], [-1,8,-1], [-1,-1,-1]])
    kernel = torch.stack((k,k,k), dim=0).unsqueeze(dim=0) # Add an outer-most dimension
    print(f"Kernel shape: {kernel.shape}")

    fixed_conv2d = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, padding=1, bias=False)
    fixed_conv2d.weight.data = kernel
    fixed_conv2d.weight.requires_grad = False
    laplace = fixed_conv2d(img)
    # d2l.plt.imshow(laplace.reshape(img.shape[2], img.shape[3]).detach().numpy(), cmap='gray'); d2l.plt.show()
    # Not much to see in this image, so let's try normalizing

    normalized = F.normalize(laplace, p=1, dim=1)
    min_val = torch.min(laplace)
    max_val = torch.max(laplace)
    manually_normalized = (laplace - min_val) / (max_val - min_val)
    d2l.set_figsize((7,10))
    # d2l.plt.imshow(normalized.reshape(img.shape[2], img.shape[3]).detach().numpy(), cmap='gray'); d2l.plt.show()
    # d2l.plt.imshow(manually_normalized.reshape(img.shape[2], img.shape[3]).detach().numpy(), cmap='gray'); d2l.plt.show()
    
    # Not great; let's try thresholding
    # Define a threshold value
    threshold = 0.51  # This is an example value; you may need to adjust it

    # Apply threshold
    thresholded_laplace = (manually_normalized > threshold).float() * 1  # Multiplied by 1 to convert from boolean to float
    # d2l.plt.imshow(thresholded_laplace.reshape(img.shape[2], img.shape[3]).detach().numpy(), cmap='gray'); d2l.plt.show()

    # Back to the lab
    net = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2),
        nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, padding=1),
    )
    print(net)

    def visualize_tensor(x):
        x = x.squeeze(0).permute(1,2,0)
        d2l.plt.imshow(x.detach().numpy()); d2l.plt.show()

    output = net(img)
    print(output.shape)
    # visualize_tensor(output)

    
    net = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2),
        nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, padding=1),
    )
    # torch.nn.init.xavier_uniform_(net[0].weight)
    # torch.nn.init.xavier_uniform_(net[1].weight)

    # This is the "itentity network," which learns to recreate the input image
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, weight_decay=0.0001, momentum=0.9)
    criterion = nn.MSELoss()
    for i in range(0,501):
        out = net(img)
        optimizer.zero_grad()
        loss = criterion(out, img)
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            print(f"Epoch {i}: Loss {loss.item()}")

    # Pretty good
    # visualize_tensor(net(img))

    # Now let's try a different image
    new_cat = d2l.Image.open("data/d2l/grumpycat.jpg")
    new_cat = TF.resize(new_cat, (400, 500))
    new_cat = TF.to_tensor(new_cat).reshape(1, 3, 400, 500)
    visualize_tensor(net(new_cat))
