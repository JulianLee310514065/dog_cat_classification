import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, RocCurveDisplay, roc_curve, auc

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, transforms

import timm
import argparse


# Set argparse
parser = argparse.ArgumentParser(description="Test")
parser.add_argument('-filepath', type=str, default='train', help='File path')
parser.add_argument('-batchsize', type=int, default=64, help='Set batch size')
parser.add_argument('-modelname', type=str, default='efficientnet_b0', help='Set model')
args = parser.parse_args()

# Print Argument
print('========================================')
print('Batchsize: ', args.batchsize)
print('Model Name: ', args.modelname)
print('File path: ', args.filepath)
print('========================================')



# Check category
cat_list = glob.glob(args.filepath + '/cat*.jpg')
dog_list = glob.glob(args.filepath + '/dog*.jpg')
print('Test cat images: ', len(cat_list), ' Test dog images: ',len(dog_list))
if len(cat_list) == 0 or len(dog_list) == 0:
    raise ValueError('Not found any photos')

# Check testing images 
test_list = cat_list + dog_list
print('Test all images: ', len(test_list))


# Testing dataset
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
# Testing dataset
test_dataset = CustomImageDataset(test_list, transform= trans_comp)
test_dl = DataLoader(test_dataset, batch_size= batch_size, shuffle= False, drop_last=False)
test_ld_len = len(test_dl)


# Check device
device = torch.device('cuda:0')
print('\nUse device: ', device)

# Create model
m = timm.create_model(args.modelname, pretrained=False)
class Network(nn.Module):
    def __init__(self, outdim = 1, fc1= 128, model = m):
        super(Network, self).__init__()
        
        self.model_main = model
        # FC
        
        self.fc1num = fc1
        self.fc1 = nn.Linear(1000, self.fc1num)
        self.fc2 = nn.Linear(self.fc1num, outdim)

        self.sig = nn.Sigmoid()

        self.drop = nn.Dropout(0.3)


    def forward(self, input1):
        
        output = self.model_main(input1)
                
        con = self.fc1(output)
        con = self.drop(con)
        con = self.fc2(con)   
        con = self.sig(con)

        return con
    
# Load model
model = Network()
print('Load model ', model.load_state_dict(torch.load("cat_dog.pth")))
model.to(device)

# Set loss function
loss_function = nn.BCELoss()

# Test
model.eval()
acc = 0.
epoch_loss = 0.
output_list = np.array([])
labels_list = np.array([])

for batch, (images, labels) in enumerate(test_dl):

    print('Testing', batch + 1, '/', test_ld_len ,end='\r')

    inputs, labels = images.to(device), labels.float().to(device) 
    outputs = model(inputs).squeeze() #forward    
    loss = loss_function(outputs, labels)
    epoch_loss += float(loss.item())


    output_list = np.append(output_list, outputs.cpu().detach().numpy())
    labels_list = np.append(labels_list, labels.cpu().detach().numpy())
    
predict_list = np.where(output_list> 0.5, 1, 0)

avg_test_acc = sum(predict_list == labels_list)/predict_list.size
avg_test_loss = epoch_loss/(batch + 1)

print('Test batchs: ', (batch + 1))
print('Test Ave accuracy: ', avg_test_acc)  
print('Test Loss: ', avg_test_loss)


# Classification report
print("\n", classification_report(labels_list, predict_list,  digits= 4))

# Confusion matric
conf_num = confusion_matrix(labels_list, predict_list)
conf_per = np.array([(x/sum(x)) for x in conf_num])
print('Confusion Metric: \n', conf_num)
print('Confusion Metric (percent): \n', conf_per)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize= (10, 5))
sns.set(font_scale=1.5)
sns.heatmap(conf_num , annot= True, cmap='Blues', ax= ax1, fmt='.0f')
ax1.set_title("Confusion matric")
ax1.set_xticks([0.5, 1.5], ['Benign', 'Malignant'])
ax1.set_yticks([0.5, 1.5], ['Benign', 'Malignant'])

sns.heatmap(conf_per , annot= True, cmap='Blues', ax= ax2, fmt='.3f')
ax2.set_title("Confusion matric")
ax2.set_xticks([0.5, 1.5], ['Benign', 'Malignant'])
ax2.set_yticks([0.5, 1.5], ['Benign', 'Malignant'])
plt.tight_layout()


# ROC curve
fpr, tpr, thresholds = roc_curve(labels_list, output_list)
roc_auc = auc(fpr, tpr)
display  = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
display.plot()
ax_roc = plt.gca()
ax_roc.set_title('ROC curve')

plt.show()