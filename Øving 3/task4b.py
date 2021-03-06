import pathlib
import matplotlib.pyplot as plt
from PIL import Image
import torchvision
import torch
import numpy as np
image_og = Image.open("images/zebra.jpg")
print("Image shape:", image_og.size)

model = torchvision.models.resnet18(pretrained=True)
print(model)
first_conv_layer = model.conv1
print("First conv layer weight shape:", first_conv_layer.weight.shape)
print("First conv layer:", first_conv_layer)

# Resize, and normalize the image with the mean and standard deviation
image_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
image = image_transform(image_og)[None]
print("Image shape:", image.shape)

activation = first_conv_layer(image)
print("Activation shape:", activation.shape)


def torch_image_to_numpy(image: torch.Tensor):
    """
    Function to transform a pytorch tensor to numpy image
    Args:
        image: shape=[3, height, width]
    Returns:
        iamge: shape=[height, width, 3] in the range [0, 1]
    """
    # Normalize to [0 - 1.0]
    image = image.detach().cpu() # Transform image to CPU memory (if on GPU VRAM)
    image = image - image.min()
    image = image / image.max()
    image = image.numpy()
    if len(image.shape) == 2: # Grayscale image, can just return
        return image
    assert image.shape[0] == 3, "Expected color channel to be on first axis. Got: {}".format(image.shape)
    image = np.moveaxis(image, 0, 2)
    return image

plot_path = pathlib.Path("plots")
#Task 4b plotting
def plot_4b():
    indices = [14, 26, 32, 49, 52]

    plt.figure(figsize=(15, 6))
    for i in range(len(indices)):
        #Plotting filter
        plt.subplot(2,len(indices), i + 1)
        kernel = torch_image_to_numpy(first_conv_layer.weight[indices[i],:,:,:])
        plt.imshow(kernel)
        plt.title(f"Filter {indices[i]}")

        #Plotting activation from corresponding filter
        plt.subplot(2,len(indices),i+len(indices) + 1)
        act_im = torch_image_to_numpy(activation[0,indices[i],:,:])
        plt.imshow(act_im, cmap='gray')
        plt.title(f"Activation {indices[i]}")
    
    plt.savefig(plot_path.joinpath(f"task4b_activations.png"))
    plt.show()


#4c plotting
def plot_4c():
    #Passing the image through all but the last two modules
    img = model.conv1(image)
    img = model.bn1(img)
    img = model.relu(img)
    img = model.maxpool(img)
    img = model.layer1(img)
    img = model.layer2(img)
    img = model.layer3(img)
    img = model.layer4(img)

    #Plotting activations from the 10 first filters
    plt.figure(figsize=(15,6))
    for i in range(10):
        plt.subplot(2,5,i+1)
        act_im = torch_image_to_numpy(img[0,i,:,:])
        plt.imshow(act_im)
        plt.title(f"Activation {i+1}")
    plt.savefig(plot_path.joinpath(f"task4c_activations.png"))
    plt.show()


plot_4b()
plot_4c()
