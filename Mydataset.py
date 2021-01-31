import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
DATASET_PATH='E:\LearningStuff\DLcode\Pytorch\Mnist\datasets'
def get_train_data_loader(batch_size):
    train_set=torchvision.datasets.MNIST(
        root=DATASET_PATH,
        train=True,
        download=False,
        transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(0.1307,0.3081)])    
    )
    return torch.utils.data.DataLoader(train_set,batch_size=batch_size,shuffle=True)

def get_test_data_loader(batch_size):
    test_set=torchvision.datasets.MNIST(
        root=DATASET_PATH,
        train=False,
        download=False,
        transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(0.1307,0.3081)])    
    )
    return torch.utils.data.DataLoader(test_set,batch_size=batch_size,shuffle=True)

class ExpandingDataset(Dataset):
    def __init__(self,numbers,length):
        '''在原有的60000张训练集图片中，添加额外的图片'''
        '''numbers为需要扩展的数字，length为扩展长度,每个数字均匀扩展'''
        train_loader=get_train_data_loader(60000)
        batch=next(iter(train_loader))
        images,labels=batch
        each_length=length//len(numbers)
        for num in numbers:
            index=(labels==num)
            temp_im=images[index]
            temp_im=temp_im[0:each_length]
            images=torch.cat((images,temp_im),0)
            labels=torch.cat((labels,num*torch.ones((each_length),dtype=torch.int64)))
        self.expset=[images,labels]
        self.transform=transform
    def __getitem__(self,index):
        image=self.expset[0][index]
        label=self.expset[1][index]
        return image,label
    def __len__(self):
        return len(self.expset[0])

def get_expanding_data_loader(numbers,length,batch_size):
    expanding_set=ExpandingDataset(numbers=numbers,length=length)
    return torch.utils.data.DataLoader(expanding_set,batch_size=batch_size,shuffle=True)
