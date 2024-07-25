import os
import copy
import math
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
#from sklearn.feature_extraction import image #reconstract from patches 2d
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as TF

import torchvision.utils
from collections import defaultdict
import tifffile
import itertools


LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)
PATCH_SIZE = 32
STRIDE = 1
BATCH_SIZE = 45000
NUM_EPOCHS = 5
NUM_WORKERS = 2
IMAGE_HEIGHT =  1043 #1043 
IMAGE_WIDTH = 981
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "/home/jessica/Project-Code/x-ray-diffraction/diffraction_data/train_image"
TRAIN_MASK_DIR = "/home/jessica/Project-Code/x-ray-diffraction/diffraction_data/train_label"
VALIDATION_IMG_DIR = "/home/jessica/Project-Code/x-ray-diffraction/diffraction_data/val_image"
VALIDATION_MASK_DIR = "/home/jessica/Project-Code/x-ray-diffraction/diffraction_data/val_label"


"""
    When looking at the model for how a UNET architecture is set up, we see that for all of the "steps" the image gets 
    'transformed' two times. This is called a convolution.
        CONVOLUTION: a multiplication that is performed between an array of input data and a two-dimensional array of weights

    This is the function that creates the double convolution that brings, as describes in the uNet_arctutrecture png that moves the information, in one group
    from the left to the right.

    nn.Conv2d((in_channels, out_channels, kernel_size, stride, padding, bias)
        - in_channels: the number of channels in the input image, (colored images are 3, black and white is 1(only color in 1 dimension))
        - out_channels: number of channels produced by the convolution
        - kernel_size: widthxheight of the mask (this moves over the image) an int means that the kernal matrix is a nxn matrix and a tuple means a nxm
        - stride: how far the kernal moves each time is does, 1 mean it moves over to the next pixel
        - padding: a 1 means "same" the input hight and width will be the smae after the covolution
        - bias: do we want a learnable bias (do we want to keep the same value throughout the model?)
"""
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels), #batchnorm cancels out the bias from conv2d so we can prevent a "unless" parameter by setting bias = False
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    """
        This is the function that actually builds the movment up and down that is needed for analysing and doing the Neural Network stuff that we want
    """
    def __init__(self, in_channels=1, out_channels=1, features=[8,16]):
        super(UNET, self).__init__()

        #we want to be able to do model.eval and for the batch normal layers so that is why we choe nn.ModuleList
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature)) #matches some input to the next size ---> the first time this ever runs it matches 3 to 64
            in_channels = feature #this changes the number of inputs so we have a 1 --> 64 --> 128 situtaion so the model looks correct

        # Up part of UNET
        for feature in reversed(features):
            #this is the movement from left to right via the gray arrow in which the in_channels on the right side are double the size of the number of features on the left
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2, #the kernel_size being 2 doubles the size (heigh and width) of the image
                )
            )
            self.ups.append(DoubleConv(feature*2, feature)) #this because in each "group" we move over 2 and up one
       
        #bottleneck (the turning point from down to up)
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        #final convolution on the top right which keeps the same size of the image but decreaes the out_channels which would provide our final predicted result
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        self.final_relu = nn.ReLU() 

    def forward(self, x):
        skip_connections = []
        
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x) #the ordering is important here as we move sideways from lowest resolution first than to highest resultion

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2): # up than double conv
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2] #we have a step of 2 and we want a linear step of 1 ordering

            #if the two are not the same size, this is important since otherwise it will not work, we will take out height and width
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip) #double convolutopm

        x = self.final_conv(x)
        return self.final_relu(x)
    
class CarDataset(Dataset):
    def __init__(self, image_dir, mask_dir, patch_size, stride, transform = None, patches = None):
        self.image_dir   = image_dir
        self.mask_dir    = mask_dir
        self.transform   = transform
        self.patches     = patches
        self.patch_size  = patch_size
        self.stride      = stride
        #list all files that are in that folder
        self.images      = os.listdir(image_dir)

    #get how many images are in a specified directory
    def __len__(self):
        return len(self.images)
    
    #get a specified item in images at a specified index
    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.images[index])
        mask_path  = os.path.join(self.mask_dir, self.images[index])
        
        #loading the images
        image = np.array(Image.open(image_path).convert("L"), dtype=np.float32)
        mask  = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)

        image = image[0:256, 159:159+256]
        mask = mask[0:256, 159:159+256]

        if self.transform is not None:
            augumentation = self.transform(image = image, mask = mask)
            image = augumentation["image"]
            mask = augumentation["mask"]


        if self.patches is not None:
            image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
            mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)
            image = image.unfold(2, self.patch_size, self.stride).unfold(3, self.patch_size, self.stride)
            mask  = mask.unfold(2, self.patch_size, self.stride).unfold(3, self.patch_size, self.stride)

        return [image, mask] 
    
ALPHA = 0.8
GAMMA = 2

class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=ALPHA, gamma=GAMMA, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #first compute binary cross-entropy 
        BCE = F.binary_cross_entropy_with_logits(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE
                       
        return focal_loss
    
def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

train_transform = A.Compose(
        [
            A.Normalize(
                mean=[0.0],
                std=[1.0],
            ),
        ],
    )

val_transforms = A.Compose(
        [
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
            ),
        ],
    )

train_ds = CarDataset(image_dir=TRAIN_IMG_DIR, mask_dir=TRAIN_MASK_DIR, transform= train_transform, patches = True, patch_size=PATCH_SIZE, stride = STRIDE)
val_ds = CarDataset(image_dir=VALIDATION_IMG_DIR,mask_dir=VALIDATION_MASK_DIR, transform=val_transforms, patches = True, patch_size=PATCH_SIZE, stride = STRIDE)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, shuffle=True)
val_loader = DataLoader(val_ds,batch_size=BATCH_SIZE,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY,shuffle=False)

model = UNET(in_channels=1, out_channels=1).to(DEVICE) #if we wanted multisegmentation, we would change the number of output channels to the correct number of "classes"
#=================================================================
torch.cuda.empty_cache()
# del model
#=================================================================


#binary cross entropy - we choose this beacuse we are not doing sigmoid for the output  # Example weights: 0.1 for background, 0.9 for areas of interest
# loss_fn = nn.BCELoss()  
loss_fn = FocalLoss() #we would change this to CEEithLogitsLoss() if multi classes
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scaler = torch.cuda.amp.GradScaler()
print(model)


if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

#train the model
def train_fn(train_data, model, optimizer, loss_fn, scaler, device):
    model.train()
    loop = tqdm(train_data)
    running_loss = 0.0
    for _, (image_batch, mask_batch) in enumerate(loop):
        for batch_image, batch_mask in zip(image_batch, mask_batch):
            batch_image = batch_image.contiguous().view(-1, PATCH_SIZE, PATCH_SIZE)
            batch_mask = batch_mask.contiguous().view(-1, PATCH_SIZE, PATCH_SIZE)

            for img_patch, mask_patch in zip(batch_image, batch_mask):
                img_patch = img_patch.unsqueeze(0).unsqueeze(0).float().to(device)
                mask_patch = mask_patch.unsqueeze(0).unsqueeze(0).to(device)

                with torch.cuda.amp.autocast():
                    # img_patch = nor_data(img_patch) #use pytorch's instead
                    #convert back to tensor
                    # print("img: " + str(img_patch))

                    predicted = model(img_patch)
                    # print(predicted)
                    #only for use in BCECrossEntropyLogit
                    # predicted = torch.sigmoid(predicted)
                    loss = loss_fn(predicted, mask_patch)
                    # print("loss: " + str(loss))

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item()

                loop.set_postfix(loss=loss.item())

    average_loss = running_loss / len(train_data)
    print(f'Average loss: {average_loss}')
    return average_loss

def accuracy_of_model(loader, model, device="cuda"):
    model.eval()
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    with torch.no_grad():
        for _, (batch_of_images, batch_of_masks) in enumerate(loader):
            for batch_image, batch_mask in zip(batch_of_images, batch_of_masks):
                batch_image = batch_image.contiguous().view(-1, PATCH_SIZE, PATCH_SIZE)
                batch_mask = batch_mask.contiguous().view(-1, PATCH_SIZE, PATCH_SIZE)

                for img_patch, mask_patch in zip(batch_image, batch_mask):
                    img_patch = img_patch.unsqueeze(0).unsqueeze(0).float().to(device)
                    mask_patch = mask_patch.unsqueeze(0).unsqueeze(0).to(device)

                    preds = torch.sigmoid(model(img_patch))
                    preds = (preds > 0.5).float()

                    num_correct += (preds == mask_patch).sum()
                    num_pixels += torch.numel(preds)
                    dice_score += (2 * (preds * mask_patch).sum()) / ((preds + mask_patch).sum() + 1e-8)
            print(f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f} for image group")
            print(f"Dice score for group: {dice_score/len(loader)}")
    accuracy = num_correct / num_pixels * 100
    dice_score = dice_score / len(loader)
    return accuracy, dice_score
        
def get_original_image_size(patch_tensor, patch_size, stride):
    _, num_patches_h, num_patches_w, _, _ = patch_tensor.size()
    original_height = (num_patches_h - 1) * stride + patch_size
    original_width = (num_patches_w - 1) * stride + patch_size
    # if original_height == IMAGE_HEIGHT and original_width == IMAGE_WIDTH:
    #     return original_height, original_width 
    # else:
    #     raise Exception("Something went wrong, the reconstructed height and/or width does not match original demensions") 
    return original_height, original_width


def save_predictions_as_images(loader, model, patch_size, stride, device="cuda"):
    model.eval()
    processed_patches = []
    for _, (batch_of_images, batch_of_masks) in enumerate(loader):
        for batch_image, batch_mask in zip(batch_of_images, batch_of_masks):
            processed_patches_per_image = []
            batch_image = batch_image.contiguous().view(-1, patch_size, patch_size) #makes into 
            batch_mask = batch_mask.contiguous().view(-1, patch_size, patch_size)
            for img_patch, mask_patch in zip(batch_image, batch_mask):
                img_patch = img_patch.unsqueeze(0).unsqueeze(0).float().to(device)
                mask_patch = mask_patch.unsqueeze(0).unsqueeze(0).to(device)
                with torch.no_grad():
                    preds = torch.sigmoid(model(img_patch))
                    preds = (preds > 0.5).float()
                processed_patches_per_image.append(preds) #per each patch for a specific image
            processed_patches.append(torch.stack(processed_patches_per_image)) #combines all patches to a single tensor, each index of the array is a seperate image.
    for index, image in enumerate(processed_patches):
        image = image.squeeze(1).squeeze(1)

        #reshape to get back original size
        num_patches = image.shape[0]
        # Calculate the dimensions of the original image
        image_dim = int(math.sqrt(num_patches))
        # Reshape to [1, 1, image_dim, image_dim, patch_size, patch_size]
        reshaped_tensor = image.view(1, image_dim, image_dim, patch_size, patch_size)
        original_image_size = get_original_image_size(reshaped_tensor, patch_size, stride)
        reconstruct_patches(image.cpu().numpy(), original_image_size, stride, index)
    
    model.train()
    return

def reconstruct_patches(patches, image, stride, index):
    """
        Reconstruct the image from all of its patches.
        Patches are assumed to overlap and the image is constructed by filling in
        the patches from left to right, top to bottom, averaging the overlapping
        regions.

        Parameters
        ----------
        patches : array, shape = (n_patches, patch_height, patch_width) or
            (n_patches, patch_height, patch_width, n_channels)
            The complete set of patches. If the patches contain colour information,
            channels are indexed along the last dimension: RGB patches would
            have `n_channels=3`.

        image_size : tuple of ints (image_height, image_width) or
            (image_height, image_width, n_channels)
            the size of the image that will be reconstructed

        step: number of pixels between two patches

        Returns
        -------
        image : array, shape = image_size
            the reconstructed image
        """
    i_h, i_w = image[:2]
    p_h,p_w = patches.shape[1:3]
    img = np.zeros(image)
    # compute the dimensions of the patches array
    n_h = (i_h - p_h) // stride + 1
    n_w = (i_w - p_w) // stride + 1
    # Get the Cartesian product of the grid indices 
    indices = list(itertools.product(range(n_h), range(n_w))) 
    # Iterate over the indices and patches
    for idx, (i, j) in enumerate(indices):
        p = patches[idx] 
        img[i * stride:i * stride + p_h, j * stride:j * stride + p_w] += p
    for i in range(i_h):
        for j in range(i_w):
            img[i, j] /= float(min(i + stride, p_h, i_h - i) *
                                min(j + stride, p_w, i_w - j))
    str_test = "test_" + str(index) + ".tiff"
    # Plot the reconstructed image
    tifffile.imwrite(str_test, img.astype(np.float32) )
    return 

# accuracy_of_model(val_loader, model, device = DEVICE)
# for epochs in range(NUM_EPOCHS): #how many times do we want to train?
#     # save model
#     checkpoint = {
#         "state_dict": model.state_dict(),
#         "optimizer":optimizer.state_dict(),
#     }
#     save_checkpoint(checkpoint)
#     train_fn(train_loader, model, optimizer, loss_fn, scaler, device = DEVICE)
#     accuracy_of_model(val_loader, model, device = DEVICE)
# save_predictions_as_images(val_loader, model = model, patch_size = PATCH_SIZE, stride = STRIDE, device=DEVICE)

def main():
    print("enter main")
    train_losses = []
    val_accuracies = []
    val_dice_scores = []

    accuracy_of_model(val_loader, model, device = DEVICE)
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch [{epoch}/{NUM_EPOCHS}]")
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)
        # Train
        print(DEVICE)
        train_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler, DEVICE)
        train_losses.append(train_loss)

        # Validate
        val_accuracy, val_dice_score = accuracy_of_model(val_loader, model, DEVICE)
        val_accuracies.append(val_accuracy.cpu())
        val_dice_scores.append(val_dice_score)

        print(f"Validation Accuracy: {val_accuracy:.2f}%")
        print(f"Dice Score: {val_dice_score:.4f}")

    save_predictions_as_images(val_loader, model = model, patch_size = PATCH_SIZE, stride = STRIDE, device=DEVICE)

    print(train_losses, val_accuracies, val_dice_scores)
    # fig, axes = plt.subplots(1, 1, figsize=(10, 15))
    plt.figure()
    plt.plot(range(NUM_EPOCHS), train_losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Time")
    plt.legend()
    plt.show()
    
    # axes[0].plot(range(0, NUM_EPOCHS), train_losses, label="Training Loss")
    # axes[0].set_xlabel("Epoch")
    # axes[0].set_ylabel("Loss")
    # axes[0].set_title("Training Loss Over Time")
    # axes[0].legend()

    # axes[1].plot(range(NUM_EPOCHS), val_accuracies, label="Validation Accuracy")
    # axes[1].set_xlabel("Epoch")
    # axes[1].set_ylabel("Accuracy (%)")
    # axes[1].set_title("Validation Accuracy Over Time")
    # axes[1].legend()

    # axes[2].plot(range(NUM_EPOCHS), val_dice_scores, label="Dice Score")
    # axes[2].set_xlabel("Epoch")
    # axes[2].set_ylabel("Dice Score")
    # axes[2].set_title("Validation Dice Score Over Time")
    # axes[2].legend()

    # plt.tight_layout()
    # plt.show()


if __name__ == "__main__":
    main()