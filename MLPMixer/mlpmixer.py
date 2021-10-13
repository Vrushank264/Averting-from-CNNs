#In Token Mixing I used traditional transpose but in Channel mixing i tried Rearrange from "Einops" library 
#Einops Rearrange Docs: https://einops.rocks/1-einops-basics/ 
#Token Mixing: Rearrange('b h w c -> h (b w) c')
#Channel Mixing: Rearrange('b c h w -> b (h w) c')
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from dataset import dataset_Loader
from sklearn.metrics import precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import gc
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler
import time
import config


def flatten_list(_2d_list):
    flat_list = []
    # Iterate through the outer list
    for element in _2d_list:
        if type(element) is list:
            # If the element is of type list, iterate through the sublist
            for item in element:
                flat_list.append(item)
        else:
            flat_list.append(element)
    return flat_list

class MLPblock(nn.Module):
    def __init__(self, in_c, out_c):
        super(MLPblock, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_c, out_c),
            nn.GELU(),
            nn.Linear(out_c, in_c)
            )
    
    def forward(self, x):
        return self.mlp(x)
    
class MixerLayer(nn.Module):
    def __init__(self, tokens, hidden_dim, token_mix_dim, channel_mix_dim):
        super(MixerLayer, self).__init__()
        self.layernorm = nn.LayerNorm(hidden_dim)
        self.token_mixing = MLPblock(tokens, token_mix_dim)
        self.channel_mixing = MLPblock(hidden_dim, channel_mix_dim)
        
    def forward(self, x):
        out = self.layernorm(x).transpose(1,2)
        x = x + self.token_mixing(out).transpose(1,2)
        out = self.layernorm(x)
        x = x + self.channel_mixing(out)
        return x
    
class MLPMixer(nn.Module):
    def __init__(self, img_size, patch_size, n_blocks, hidden_dim, token_mix_dim,channel_mix_dim, n_classes):
        super(MLPMixer, self).__init__()
        assert (img_size % patch_size) == 0
        tokens = (img_size // patch_size)**2
        self.patch_embedding = nn.Sequential(nn.Conv2d(in_channels=3,
                                         out_channels=hidden_dim,
                                         kernel_size = patch_size,
                                         stride=patch_size,
                                         bias = False),
                                             Rearrange('b c h w -> b (h w) c')
                                             )
        self.mixer_blocks = nn.ModuleList([])
        for _ in range(n_blocks):
            self.mixer_blocks.append(MixerLayer(tokens, hidden_dim, token_mix_dim, channel_mix_dim))
        self.layernorm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, n_classes)
        
        self._initialize_weights()
        
    def forward(self, x):
        x = self.patch_embedding(x)
        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)
            
        x = self.layernorm(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                nn.init.normal_(m.bias.data)
    
    
def train(model, train_dataloader, val_dataloader, loss_fn, optimizer, device, epochs, path, tb, sch):
    for epoch in range(1,epochs+1):
        train_losses = 0.0
        val_losses = 0.0
        train_accuracy_array = []
        val_accuracy_array = []
        print('\nTraining...')
        for image, target in tqdm(train_dataloader,):
            image, target = image.to(device), target.to(device)
            
            predictions = model(image)
            
            loss = loss_fn(predictions, target)
            train_losses += loss.item()*image.size(0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            
            _, preds = torch.max(predictions.detach(), dim=1)
            accuracy = torch.tensor(torch.sum(preds == target).item()) / len(preds)
            train_accuracy_array.append(accuracy)
            
            if epoch == 25:
                torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, path)       

        train_loss = train_losses/len(train_dataloader.dataset)
        tb.add_scalars('Loss', {'Train Loss':train_loss}, epoch)
        tb.add_scalars('Accuracy', {'Train Accuracy':np.mean(train_accuracy_array)}, epoch)
        
        print('\nValidating...')
        
        model.eval()
        for image, target in tqdm(val_dataloader):
            with torch.no_grad():
                image, target = image.to(device), target.to(device)
                
                preds = model(image)
                valid_loss = loss_fn(preds, target)
                val_losses += valid_loss.item()*image.size(0)
                
                _, pred = torch.max(preds, 1)
                accuracy = torch.tensor(torch.sum(pred == target).item()) / len(pred)
                val_accuracy_array.append(accuracy)
        
        sch.step(val_losses)
                
        val_loss = val_losses/len(val_dataloader.dataset)
        print(f"\nEpoch : {epoch}\nTrain Loss: {train_loss}     Train Accuracy:{np.mean(train_accuracy_array)}") 
        print(f"Validation Loss: {val_loss}   Validation Accuracy:{np.mean(val_accuracy_array)}") 
        tb.add_scalars('Loss', {'Validation Loss':val_loss}, epoch)
        tb.add_scalars('Accuracy', {'Validation Accuracy':np.mean(val_accuracy_array)}, epoch)
        
        model.train()
            
        gc.collect()
        torch.cuda.empty_cache()
        

def test(model, test_dataloader, loss_fn, optimizer, device):
    test_loss = 0.0
    class_correct = list(0. for i in range(3))
    class_total = list(0. for i in range(3))
    classes = ['COVID19', 'NORMAL', 'PNUEMONIA']
    y_preds = []
    y_actual = []
    
    model.eval()
    for image, target in tqdm(test_dataloader):
        with torch.no_grad():
            image, target = image.cuda(), target.cuda()
            output = model(image)
            loss = loss_fn(output, target)
            test_loss += loss.item()*image.size(0)
            _, pred = torch.max(output, 1)    
            correct_tensor = pred.eq(target.data.view_as(pred))
            correct = np.squeeze(correct_tensor.cpu().numpy())
            for i in range(0, len(image)):
                label = target.data[i]
                class_correct[label] += correct[i].item()
                class_total[label] += 1
            
            prediction = pred.cpu().numpy()
            real_tar = target.cpu().numpy().tolist()
            prediction = prediction.round().tolist()
            y_preds.append(prediction)
            y_actual.append(real_tar)
    
    y_preds = flatten_list(y_preds)
    y_actual = flatten_list(y_actual)
    y_preds, y_actual = np.vstack(y_preds), np.vstack(y_actual)

    precision = precision_score(y_actual, y_preds, average='macro')
    recall = recall_score(y_actual, y_preds, average='macro')
    confusion_m = confusion_matrix(y_actual, y_preds)
    
    # average test loss
    test_loss = test_loss/len(test_dataloader.dataset)
    print('\nTest Loss: {:.6f}\n'.format(test_loss))
    
    print(f"Precision:{precision}, Recall:{recall}")
    
    df_cm = pd.DataFrame(confusion_m, classes, classes)
    sns.heatmap(df_cm, annot=True, fmt='g')
    plt.show()
    
    
    for i in range(3):
        if class_total[i] > 0:
            print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                classes[i], 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])))
    
    print('\nTest Accuracy (Overall): %2f%% (%2d/%2d)' % (
        100. * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total)))
            

def test_throughput(model, test_dataloader, device):
  model.eval()
  start = time.time()
  for data, target in tqdm(test_dataloader):
    data, target = data.to(device), target.to(device)
  time_taken = time.time() - start
  throughput = len(test_dataloader.dataset)/time_taken
  return throughput


if __name__ == '__main__':
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    train_dir = config.TRAIN_DIR
    test_dir = config.TEST_DIR
    
    train_dataloader, val_dataloader = dataset_Loader(train_dir, batch_size=16, train=True)
    
    test_dataloader = dataset_Loader(test_dir, batch_size=16,train=False)
    
    # img_size, patch_size, n_blocks, hidden_dim, token_mix_dim,channel_mix_dim, n_classes
    model = MLPMixer(256, 32, 8, 512, 256, 2048, 3).to(device)
    
    tb = SummaryWriter('runs/MlpMixerB32')
    
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('\nTrainable Parameters: %.3fM' % parameters)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    
    lr_sch = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, verbose=True)
    
    if config.SAVED_MODEL:
        path = config.SAVED_MODEL_PATH
        
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
    
    save_path = '.\\data\\mlpMixerB16.pt'
    
    start_time = time.time()
    train(model, train_dataloader, val_dataloader, loss_fn, optimizer, device, 1, save_path, tb, lr_sch)
    end_time = time.time() - start_time

    print('Total Time : ', str(end_time))
    test(model, test_dataloader, loss_fn, optimizer, device)
    throughput = test_throughput(model, test_dataloader, device)
    print(throughput)
