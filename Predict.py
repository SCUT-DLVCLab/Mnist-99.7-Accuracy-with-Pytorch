import Mydataset
import torch
import torchvision
import torchvision.transforms as transforms
from cnnModel import CNN
from Mydataset import get_test_data_loader
import matplotlib.pyplot as plt
from PIL import Image
PATH="E:\LearningStuff\DLcode\Pytorch\Mnist\Trained_models"
def predict(image):
    if not torch.is_tensor(image):
        image=image.resize((28,28))  
        '''将图像转为tensor'''
        loader=transforms.Compose([transforms.ToTensor(),transforms.Normalize(0.1307,0.3081)])
        image=loader(image).unsqueeze(dim=0)
    ''' 预测 '''
    network=CNN()
    network.eval()
    network.load_state_dict(torch.load(PATH+"\Model1.pkl"))
    pred=network(image).argmax(dim=1)
    return pred

if __name__=='__main__':
    Load_from_file=True
    
    if Load_from_file:
       
        image=Image.open('Images/new2.jpg').convert('L')
        pred=predict(image)
    else :
        batch=get_test_data_loader(batch_size=1)
        image,label=next(iter(batch))
        pred=predict(image)
        print("the Prediction of ",str(label.numpy())," is:",str(pred.numpy()))
    print('The predition is:{}'.format(pred))
