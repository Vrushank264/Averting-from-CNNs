import os
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
from torchvision import transforms, datasets


def dataset_Loader(root_dir, batch_size, train=True):
    if train:
        transform = transforms.Compose([transforms.Resize((256, 256)),
                                        transforms.ToTensor()])
        
        dataset = datasets.ImageFolder(root_dir, transform=transform)
        
        train_dataset, val_dataset = random_split(dataset, [len(dataset)-round(0.15*(len(dataset))), round(0.15*(len(dataset)))])
        
        train_class_weights = []
        for root, subdir, files in os.walk(root_dir):
            if len(files) > 0:
                train_class_weights.append(1/len(files))
                
        train_sample_weights = [0] * len(train_dataset)
                
        for idx, (data, label) in enumerate(train_dataset):
            class_weight = train_class_weights[label]
            train_sample_weights[idx] = class_weight
        
        train_sampler = WeightedRandomSampler(train_sample_weights, num_samples = len(train_sample_weights), replacement=True)
        
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, pin_memory=True)  
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        
        return train_dataloader, val_dataloader
    else:
        transform = transforms.Compose([transforms.Resize((256, 256)),
                                        transforms.ToTensor()])
        
        dataset = datasets.ImageFolder(root_dir, transform=transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        return dataloader
 
