from d2l import torch as d2l
import numpy as np

def explore_d2l(args):
    img = d2l.Image.open("data/d2l/cat1.jpg")
    img = np.asarray(img)
    print(f"Image shape: {img.shape}")
    print(f"Image dtype: {img.dtype}")
    #print(f"Image data:\n{img}")
    #d2l.plt.imshow(img); d2l.plt.show()

    # Zero-out the Green and Blue pixels to see just eh red component
    red_img = img.copy()
    red_img[:, :, 1] = 0
    red_img[:, :, 2] = 0
    # d2l.plt.imshow(red_img); d2l.plt.show()

    blue_img = img.copy()
    blue_img[:, :, 0] = 0
    blue_img[:, :, 1] = 0
    d2l.plt.imshow(blue_img); d2l.plt.show()