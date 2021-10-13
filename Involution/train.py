import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import DataLoader
from torchvision import transforms,datasets
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
from InvNet import InvNet
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

batch_size = 32
valid_size = 0.2
num_epochs = 25
data_dir = 'E:/Computer Vision/Research/Data'
log_dir = 'E:/Computer Vision/Research/logs'


def imshow(img):
    plt.imshow(np.transpose(img, (1,2,0)))


def get_data(root, mode):
    
    if mode == 'train':
        path = os.path.join(root,'train')
    elif mode == 'test':
        path = os.path.join(root, 'test')
    else:
        raise Exception("Specify mode in get_data function!")
    
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256,256)),
    transforms.RandomRotation(degrees=15)
    ])
       
    dataset = datasets.ImageFolder(root = path, transform = transform)
    return dataset


def train(loader, epochs):
    
    for epoch in range(1, epochs + 1):
        
        model.train()
        torch.cuda.empty_cache()
        train_loss = 0.0
        valid_loss = 0.0
        train_accuracy_array = []
        val_accuracy_array = []
        for batch_idx, (images, targets) in enumerate(tqdm(train_loader,leave = True, position = 0)):
            
            images, targets = images.to(torch.device('cuda')), targets.to(torch.device('cuda'))
            outputs = model(images)
            loss = criterion(outputs, targets)
            train_loss += loss.item()*images.size(0)
            loss.backward()
            optimizer.step()
            
            _, preds = torch.max(outputs.detach(), dim = 1)
            acc = torch.tensor(torch.sum(preds = targets).item()) / len(preds)
            train_accuracy_array.append(acc)
            
            if epoch % 2 == 0:
                torch.save(model.state_dict(), '')
            
        train_loss = train_loss / len(train_loader.sampler)
        writer.add_scalars('Loss', {'Train Loss': train_loss}, epoch)
        writer.add_scalars('Accuracy', {'Train Accuracy': train_accuracy_array[epoch-1]}, epoch)
        print('\nValidating...')
        model.eval()
        for batch_idx, (images, targets) in enumerate(tqdm(valid_loader,leave = True, position = 0)):
            with torch.no_grad():
                images, targets = images.to(torch.device('cuda')), targets.to(torch.device('cuda'))
                outputs = model(images)
                loss = criterion(outputs, targets)
                valid_loss += loss.item()*images.size(0)
                
                _, pred = torch.max(outputs, 1)
                val_acc = torch.tensor(torch.sum(pred == targets).item()) / len(pred)
                val_accuracy_array.append(val_acc)
                
            
        valid_loss = valid_loss / len(valid_loader.sampler)
        
        scheduler.step(valid_loss)
        print(f'Train Loss at epoch {epoch}: {train_loss} and Accuracy: {train_accuracy_array[epoch-1]}')
        print(f'Valid Loss at epoch {epoch}: {valid_loss} and Accuracy: {val_accuracy_array[epoch-1]}')
        writer.add_scalars('Loss', {'Validation Loss': valid_loss}, epoch)
        writer.add_scalars('Accuracy', {'Valdation Accuracy': val_accuracy_array[epoch-1]}, epoch)
        
        
def check_acc(loader, model, criterion):
    
    num_correct = 0
    num_samples = 0
    model.eval()
    
    with torch.no_grad():
        for x,y in loader:
            x, y = x.to(torch.device('cuda')), y.to(torch.device('cuda'))
            scores = model(x)
            
            _, pred = scores.max(1)
            num_correct += (pred == y).sum()
            num_samples += pred.size(0)
            
        print(f'Got {num_correct} / {num_samples} with accuracy: {float(num_correct)/float(num_samples)*100:.4f}')
        

train_data = get_data(root = data_dir, mode = 'train')
test_data = get_data(root = data_dir, mode = 'test')


#Splitting training and Validation data
num_train_sample = len(train_data)
indices = list(range(num_train_sample))
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train_sample))
train_idx, valid_idx = indices[split:], indices[:split]

#Using SubsetRandomSampler to address data imbalance problem
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

train_loader = DataLoader(train_data, batch_size = batch_size, sampler = train_sampler)
valid_loader = DataLoader(train_data, batch_size = batch_size, sampler = valid_sampler)
test_loader = DataLoader(test_data, batch_size = batch_size)

classes = ['COVID19', 'NORMAL', 'PNEUMONIA']
    
dataiter = iter(train_loader)
images, labels = dataiter.next()
images = images.numpy()


fig = plt.figure(figsize = (25,4))
for idx in np.arange(20):
    ax = fig.add_subplot(2, 10, idx+1, xticks = [], yticks = [])
    imshow(images[idx])
    ax.set_title(classes[labels[idx]])


model = InvNet(50, num_classes=3).to(torch.device('cuda'))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3,betas = (0.0, 0.99))
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor = 0.5, patience = 3, verbose = True)

 
if __name__ == '__main__':
        
        writer = SummaryWriter(log_dir)
        train(train_loader, num_epochs)
        print("Checking accuracy on training set...")
        check_acc(train_loader, model)
        print("Checking accuracy on Validation set...")
        check_acc(valid_loader, model)
            
        