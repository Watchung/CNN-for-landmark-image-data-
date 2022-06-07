# -*- coding: utf-8 -*-
"""
C:\Users\Shafufu\Desktop\Huacheng Doc\HL Python Learning\HLEnv\env1

# Create class dictionary:   classes_dict = {i: j.split(".")[1] for i,j in enumerate(os.listdir(train_dir))}  
                          or classes_dict = {i.split(".")[0] : i.split(".")[1] for i in os.listdir(train_dir)} 






"""



==========Step 0: Download Datasets and Install Python Modules==========

import os
import torch
import numpy as np
from six.moves import urllib
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

os.chdir(r"C:\Users\Shafufu\Desktop\Huacheng Doc\HL Python Learning\Udacity\04 Deep Learning\Project_2_landmark\landmarks")
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler




==========Step 1: Create a CNN to Classify Landmarks (from Scratch)==========

# =================================Load data
data_dir = '/data/landmark_images/' # Use this one in Udacity train_dir = os.path.join(data_dir, 'train/') ; os.listdir(data_dir)

data_dir = 'landmark_images/'
train_dir = os.path.join(data_dir, 'train/')
test_dir = os.path.join(data_dir, 'test/')

classes_dict = {i: j.split(".")[1] for i,j in enumerate(os.listdir(train_dir))} # return name for a given label
classes = list(classes_dict.values())

normalize_mean = np.array([0.5, 0.5, 0.5])
normalize_std = np.array([0.5, 0.5, 0.5])

def data_transforms():
    data_transforms = {}
    data_transforms["train"] = transforms.Compose([
                                           transforms.RandomChoice([
                                               transforms.RandomRotation(180),
                                               transforms.RandomHorizontalFlip(p=0.5),
                                               transforms.RandomVerticalFlip(p=0.5)]),
                                           transforms.RandomResizedCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize(normalize_mean,normalize_std)])

    data_transforms["test"]  = transforms.Compose([
                                           transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize(normalize_mean,normalize_std)])
    return(data_transforms)

data_transforms = data_transforms()

def image_datasets(data_dir, data_transforms):
    image_datasets = {}
    image_datasets["train"] = datasets.ImageFolder(data_dir + '/train', transform=data_transforms["train"])
    image_datasets["test"]  = datasets.ImageFolder(data_dir + '/test' , transform=data_transforms["test"])
    return image_datasets
image_datasets = image_datasets(data_dir, data_transforms)


def dataloaders(image_datasets):
    num_train = len(image_datasets["train"])
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(0.2 * num_train))  #valid_size = 0.2
    train_idx, valid_idx = indices[split:], indices[:split]
    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)                        
    valid_sampler = SubsetRandomSampler(valid_idx)
    
    dataloaders = {}
    dataloaders["train"] = torch.utils.data.DataLoader(image_datasets["train"], batch_size=32, sampler=train_sampler)
    dataloaders["valid"] = torch.utils.data.DataLoader(image_datasets["train"], batch_size=32, sampler=valid_sampler)
    dataloaders["test"]  = torch.utils.data.DataLoader(image_datasets["test"],  batch_size=32, shuffle=False)
    return dataloaders
loaders_scratch = dataloaders(image_datasets)


# ===============================Check data loading and visualize data
import matplotlib.pyplot as plt
%matplotlib inline

# helper function to un-normalize and display an image
def imshow(img):
    img = img / 2 + 0.5  # unnormalize 因为 Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    plt.imshow(np.transpose(img, (1, 2, 0)))  # convert from Tensor image

# obtain one batch of training images
dataiter = iter(dataloaders['train'])
images, labels = dataiter.next()
images = images.numpy() # convert images to numpy for display

# plot the images in the batch, along with the corresponding labels
fig = plt.figure(figsize=(25, 4))
# display 20 images
for idx in np.arange(20):
    ax = fig.add_subplot(2, int(20/2), idx+1, xticks=[], yticks=[])
    imshow(images[idx])
    ax.set_title(classes[labels[idx]])

rgb_img = np.squeeze(images[3]); imshow(rgb_img)

# obtain one batch of training images
dataiter = iter(loaders_scratch['train'])
images, labels = dataiter.next()
images = images.numpy() # convert images to numpy for display

# plot the images in the batch, along with the corresponding labels
fig = plt.figure(figsize=(25, 4))
# display 20 images
for idx in np.arange(20):
    ax = fig.add_subplot(2, int(20/2), idx+1, xticks=[], yticks=[])
    imshow(images[idx])
    ax.set_title(classes[labels[idx]])

rgb_img = np.squeeze(images[3]); imshow(rgb_img)

# ================================Create Network
import torch.nn as nn
import torch.nn.functional as F
# define the CNN architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # convolutional layer (sees 3x224x224 image tensor)
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1) 
        # convolutional layer (sees 16x16x16 tensor)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1) # HL: 16 可以写成self.conv1.out_channels
        # convolutional layer (sees 8x8x32 tensor)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1) # HL： 32可以写成self.conv2.out_channels
        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        # linear layer (64 * 4 * 4 -> 500)
        self.fc1 = nn.Linear(64 * 28 * 28, 1000)
        # linear layer (500 -> 10)
        self.fc2 = nn.Linear(1000, 50)
        # dropout layer (p=0.25)
        self.dropout = nn.Dropout(0.25)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = self.pool(F.relu(self.conv1(x)))               
        x = self.pool(F.relu(self.conv2(x)))                
        x = self.pool(F.relu(self.conv3(x)))                
        # flatten image input
        x = x.view(-1, 64 * 28 * 28)
        # add dropout layer
        x = self.dropout(x)
        # add 1st hidden layer, with relu activation function
        x = F.rrelu(self.fc1(x))
        # add dropout layer
        x = self.dropout(x)
        # add 2nd hidden layer, with relu activation function
        x = self.fc2(x)
        x = self.logsoftmax(x)
        return x

# create a complete CNN
model = Net()
#print(model)
# useful variable that tells us whether we should use the GPU
if torch.cuda.is_available():
    train_on_gpu=True
    use_cuda=True
    device = torch.device("cuda")
    model.cuda()  # or model.to(device)
else:
    train_on_gpu=False
    use_cuda=False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# specify loss function (categorical cross-entropy)
criterion = nn.NLLLoss() 
# specify optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
# number of epochs to train the model
n_epochs = 3
valid_loss_min = np.Inf 

for epoch in range(1, n_epochs+1):
    # keep track of training and validation loss
    train_loss = 0.0
    valid_loss = 0.0
    ###################
    # train the model #
    ###################
    model.train()
    for data, target in dataloaders['train']:
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = criterion(output, target)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update training loss
        train_loss += loss.item()*data.size(0)

    ######################    
    # validate the model #
    ######################
    model.eval()
    for data, target in dataloaders['valid']:
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = criterion(output, target)
        # update average validation loss 
        valid_loss += loss.item()*data.size(0)
    
    # calculate average losses
    train_loss = train_loss/len(dataloaders['train'].sampler)
    valid_loss = valid_loss/len(dataloaders['valid'].sampler)
        
    # print training/validation statistics 
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, train_loss, valid_loss))
    
    # save model if validation loss has decreased
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
        valid_loss_min,
        valid_loss))
        torch.save(model.state_dict(), 'model_landmark_0.pt')
        valid_loss_min = valid_loss





#========================Test the Trained Network

model.load_state_dict(torch.load('model_landmark_0.pt'))
# track test loss
test_loss = 0.0
class_correct = list(0. for i in range(len(classes)))
class_total = list(0. for i in range(len(classes)))

model.eval()
# iterate over test data
for data, target in dataloaders['test']:
    # move tensors to GPU if CUDA is available
    if train_on_gpu:
        data, target = data.cuda(), target.cuda()
    # forward pass: compute predicted outputs by passing inputs to the model
    output = model(data)
    # calculate the batch loss
    loss = criterion(output, target)
    # update test loss 
    test_loss += loss.item()*data.size(0)
    # convert output probabilities to predicted class
    _, pred = torch.max(output, 1)    
    # compare predictions to true label
    correct_tensor = pred.eq(target.data.view_as(pred))
    correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
    # calculate test accuracy for each object class
    for i in range(len(target)):
        label = target.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1

# average test loss
test_loss = test_loss/len(dataloaders['test'].dataset)
print('Test Loss: {:.6f}\n'.format(test_loss))

for i in range(len(classes)):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            classes[i], 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))



















==========Step 2: Create a CNN to Classify Landmarks (using Transfer Learning)==========

import os
import numpy as np
import torch
import torchvision
from torchvision import datasets, models, transforms
from collections import OrderedDict
### TODO: Write data loaders for training, validation, and test sets
## Specify appropriate transforms, and batch_sizes

loaders_transfer = dataloaders

import torch.nn as nn
import torch.nn.functional as F

## TODO: select loss function
criterion_transfer = nn.NLLLoss() 


def freeze_parameters(root, freeze=True):
    [param.requires_grad_(not freeze) for param in root.parameters()]

def get_optimizer_transfer(model):
    ## TODO: select and return optimizer
    freeze_parameters(model)
    try: # some model final layer is called 'fc', while others are call 'classifier'
        freeze_parameters(model.fc, False)
        optimizer = torch.optim.Adagrad(model.fc.parameters(), lr=0.01, weight_decay=0.001)
    except AttributeError:
        freeze_parameters(model.classifier, False)
        optimizer = torch.optim.Adagrad(model.classifier.parameters(), lr=0.01, weight_decay=0.001)
    #optimizer = torch.optim.Adagrad(model.classifier.parameters(), lr=0.01, weight_decay=0.001)
    return(optimizer)  
  

## TODO: Specify model architecture

def customize_classifer (model_name = "vgg13", set_classifer_hiddensize=500,finalclass=50, dropout=0.5):
    model = getattr(models, model_name)(pretrained=True)   
    for param in model.parameters():
        param.requires_grad = False
    classifier_name, old_classifier = model._modules.popitem()
    try:
        old_classifier=old_classifier[0] # when model is vgg13
    except TypeError:
        pass
    classifier_input_size = old_classifier.in_features
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(classifier_input_size, set_classifer_hiddensize)),
                          ('relu', nn.RReLU()),
                          ('drop',nn.Dropout(p=dropout)),
                          ('fc2', nn.Linear(set_classifer_hiddensize,finalclass)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    model.add_module(classifier_name, classifier)
    return model

model_name='vgg16'
set_classifer_hiddensize=500
finalclass=len(classes)


model_transfer = customize_classifer (model_name = model_name,
                     set_classifer_hiddensize=set_classifer_hiddensize,
                     finalclass=finalclass)

def train(n_epochs, loaders, model, optimizer, criterion, use_cuda, save_path):
    """returns trained model"""
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf 
    
    for epoch in range(1, n_epochs+1):
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        ###################
        # train the model #
        ###################
        # set the module to training mode
        model.train()
        for batch_idx, (data, target) in enumerate(loaders['train']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()

            ## TODO: find the loss and update the model parameters accordingly
            ## record the average training loss, using something like
            ## train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data.item() - train_loss))
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update training loss
            train_loss += loss.item()*data.size(0)
        ######################    
        # validate the model #
        ######################
        # set the model to evaluation mode
        model.eval()
        for batch_idx, (data, target) in enumerate(loaders['valid']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()

            ## TODO: update average validation loss 
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # update average validation loss 
            valid_loss += loss.item()*data.size(0)

        # calculate average losses
        train_loss = train_loss/len(loaders['train'].sampler)
        valid_loss = valid_loss/len(loaders['valid'].sampler)

        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, train_loss,valid_loss ))

        ## TODO: if the validation loss has decreased, save the model at the filepath stored in save_path
        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
            torch.save(model.state_dict(), save_path)
            valid_loss_min = valid_loss        
        
    return model


if use_cuda:
    model_transfer = model_transfer.cuda()

model_transfer = train(20, loaders_transfer, model_transfer, get_optimizer_transfer(model_transfer),
                      criterion_transfer, use_cuda, 'model_transfer.pt')


#-#-# Do NOT modify the code below this line. #-#-#

# load the model that got the best validation accuracy
model_transfer.load_state_dict(torch.load('model_transfer.pt'))









==========Step 3: Write Your Landmark Prediction Algorithm==========



























