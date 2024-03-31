import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, transforms

import timm
from timm.optim import adabelief 

import argparse

# Set argparse
parser = argparse.ArgumentParser(description="Train")
parser.add_argument('-batchsize', type=int, default=64, help='Set batch size')
parser.add_argument('-modelname', type=str, default='efficientnet_b0', help='Set model')
parser.add_argument('-epochs', type=int, default=20, help='Set epoch')
parser.add_argument('-lr', type=float, default=2e-4, help='Learning rate')
parser.add_argument('-filepath', type=str, default='train', help='File path')
args = parser.parse_args()

# Print Argument
print('========================================')
print('Batchsize: ', args.batchsize)
print('Model Name: ', args.modelname)
print('Epochs: ', args.epochs)
print('Learning Rate: ', args.lr)
print('File path: ', args.filepath)
print('========================================')

# Check category
cat_list = glob.glob(args.filepath + '/cat*.jpg')
dog_list = glob.glob(args.filepath + '/dog*.jpg')
print('Cat images: ', len(cat_list), ' Dog images: ',len(dog_list))
if len(cat_list) == 0 or len(dog_list) == 0:
    raise ValueError('Not found any photos')

# Check training and validation image 
cut_num = int(len(cat_list)*0.8)
train_list = cat_list[:cut_num] + dog_list[:cut_num]
val_list = cat_list[cut_num:] + dog_list[cut_num:]
print('Train images: ', len(train_list), ' Validation images: ', len(val_list))

# Training dataset
# dog = 0, cat = 1
class CustomImageDataset(Dataset):
    def __init__(self, file_list, transform=None):

        self.file_list = file_list
        self.transform = transform    


    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):

        img_path = self.file_list[idx]        
        arr = Image.open(img_path)


        if 'dog' in img_path:
            label = 0
        elif 'cat' in img_path:
            label = 1
            

        if self.transform:
            train_img = self.transform(arr)

        
        # return image
        return train_img, label
    

# Dataset、Dataloader
batch_size = args.batchsize
# Transforms
trans_comp = transforms.Compose([transforms.Resize([128, 128]), transforms.ToTensor()])
# Training dataset
train_dataset = CustomImageDataset(train_list, transform= trans_comp)
train_dl = DataLoader(train_dataset, batch_size= batch_size, shuffle= True, drop_last=True)
train_ld_len = len(train_dl)
# Validation dataset
val_dataset = CustomImageDataset(val_list, transform= trans_comp)
val_dl = DataLoader(val_dataset, batch_size= batch_size, shuffle= False, drop_last=False)
val_ld_len = len(val_dl)

# Check device
device = torch.device('cuda:0')
print('\nUse device: ', device)

# Create model
m = timm.create_model(args.modelname, pretrained=True)
class Network(nn.Module):
    def __init__(self, outdim = 1, fc1= 128, model = m):
        super(Network, self).__init__()
        # main model
        self.model_main = model
        # FC        
        self.fc1num = fc1
        self.fc1 = nn.Linear(1000, self.fc1num)
        self.fc2 = nn.Linear(self.fc1num, outdim)
        # activation function
        self.sig = nn.Sigmoid()
        # dropout
        self.drop = nn.Dropout(0.3)

    def forward(self, input1):
        
        output = self.model_main(input1)                
        con = self.fc1(output)
        con = self.drop(con)
        con = self.fc2(con)   
        con = self.sig(con)
        return con
    
model = Network().to(device)

# Set loss function
loss_function = nn.BCELoss()

# Set optimizer
optimizer = adabelief.AdaBelief(model.parameters(), lr=args.lr, eps=1e-16, betas=(0.9, 0.99), rectify = False)


# Train and validate
pre_score = 0.
total_epochs = args.epochs
for epoch in range(total_epochs):

    print("-" * 10)
    print(f"epoch {epoch + 1}/{total_epochs}")

    # train
    model.train()
    acc = 0.
    epoch_loss = 0.
    for batch, (images, labels) in enumerate(train_dl):
           
        print('Training', batch + 1, '/', train_ld_len ,end='\r')

        inputs, labels = images.to(device), labels.float().to(device)
        optimizer.zero_grad()    
        outputs = model(inputs).squeeze() #forward    
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += float(loss.item())
        
        binary_tensor = torch.where(outputs.cpu() > 0.5, torch.tensor(1), torch.tensor(0))
        acc += sum(binary_tensor == labels.cpu()).item()/args.batchsize
        

    avg_train_acc = acc/(batch + 1)
    avg_train_loss = epoch_loss/(batch + 1)
    
    print('Train batchs: ', (batch + 1))
    print('Train Ave accuracy: ', avg_train_acc)  
    print('Train Loss: ', avg_train_loss)
    print('')
    
    # validation
    model.eval()
    acc = 0.
    epoch_loss = 0.
    val_output_list = np.array([])
    val_labels_list = np.array([])
    for batch, (images, labels) in enumerate(val_dl):
           
        print('Validating', batch + 1, '/', val_ld_len , end='\r')

        inputs, labels = images.to(device), labels.float().to(device)
        outputs = model(inputs).squeeze() #forward    
        loss = loss_function(outputs, labels)
        epoch_loss += float(loss.item())
        
        val_output_list = np.append(val_output_list, outputs.cpu().detach().numpy())
        val_labels_list = np.append(val_labels_list, labels.cpu().detach().numpy())


    val_predict_list = np.where(val_output_list> 0.5, 1, 0)
    avg_val_acc = sum(val_predict_list == val_labels_list)/val_predict_list.size
    avg_val_loss = epoch_loss/(batch + 1)

    print('')
    print('Val batchs: ', (batch + 1))
    print('Val Ave accuracy: ', avg_val_acc)  
    print('Val Loss: ', avg_val_loss)

    # save model
    if avg_val_acc > pre_score:
        pre_score = avg_val_acc
        torch.save(model.state_dict(), "cat_dog.pth")
        print('Model Saved, Accuracy: ', pre_score)
        

