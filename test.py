import torch
import torchvision
import torchvision.transforms as transforms
from ConfusuinMatrix import ConfusionMatrix

from cnnModel import CNN
from Mydataset import get_test_data_loader

path="E:\LearningStuff\DLcode\Pytorch\Mnist\Trained_models"
def new_get_num_correct(preds,labels):
    return preds.eq(labels).sum().item()

def SingleTest():
    test_loader=get_test_data_loader(batch_size=256)
    cnn=CNN()
    ''' 防止权重改变 '''
    cnn.eval() 
    cnn.load_state_dict(torch.load('newMODEL9915.pkl'))
    print("load cnn net.")
    total_correct=0
    for batchs in test_loader:
        images,labels=batchs
        preds=cnn(images)
        total_correct+=get_num_correct(preds,labels)
    print('The Accuracy of the model on test-set is:',total_correct/10000)

def CombinationTest():
    '''将九个预训练的模型级联进行预测，以达到99.7%的准确率'''
    cnn1=CNN()
    cnn1.load_state_dict(torch.load(path+"\Model1.pkl"))
    cnn1.eval()
    cnn2=CNN()
    cnn2.load_state_dict(torch.load(path+"\Model2.pkl"))
    cnn2.eval()
    cnn3=CNN()
    cnn3.load_state_dict(torch.load(path+"\Model3.pkl"))
    cnn3.eval()
    cnn4=CNN()
    cnn4.load_state_dict(torch.load(path+"\Model4.pkl"))
    cnn4.eval()
    cnn5=CNN()
    cnn5.load_state_dict(torch.load(path+"\Model5.pkl"))
    cnn5.eval()
    cnn6=CNN()
    cnn6.load_state_dict(torch.load(path+"\Model6.pkl"))
    cnn6.eval()
    cnn7=CNN()
    cnn7.load_state_dict(torch.load(path+"\Model7.pkl"))
    cnn7.eval()
    cnn8=CNN()
    cnn8.load_state_dict(torch.load(path+"\Model8.pkl"))
    cnn8.eval()
    cnn9=CNN()
    cnn9.load_state_dict(torch.load(path+"\Model9.pkl"))
    cnn9.eval()
    total_correct=0
    test_loader=get_test_data_loader(256)
    for batchs in test_loader:
        images,labels=batchs
        preds1=cnn1(images).argmax(dim=1).unsqueeze(dim=1)
        preds2=cnn2(images).argmax(dim=1).unsqueeze(dim=1)
        preds3=cnn3(images).argmax(dim=1).unsqueeze(dim=1)
        preds4=cnn4(images).argmax(dim=1).unsqueeze(dim=1)
        preds5=cnn5(images).argmax(dim=1).unsqueeze(dim=1)
        preds6=cnn6(images).argmax(dim=1).unsqueeze(dim=1)
        preds7=cnn7(images).argmax(dim=1).unsqueeze(dim=1)
        preds8=cnn8(images).argmax(dim=1).unsqueeze(dim=1)
        preds9=cnn9(images).argmax(dim=1).unsqueeze(dim=1)
        preds=torch.cat((preds1,preds2,preds3,preds4,preds5,preds6,preds7,preds8,preds9),dim=1)
        preds=preds.numpy()
        pred=[]
        for i in range(preds.shape[0]):
            temp=preds[i]
            pred.append(sorted(temp)[len(temp)//2])
        preds=torch.tensor(pred)
        total_correct+=new_get_num_correct(preds,labels)
    print("The Accuracy of the combination model on test-set is:",total_correct/10000)

def plot_confusion_matrix():
    cnn=CNN()
    cnn.load_state_dict(torch.load(path+"\Model1.pkl"))
    loader=get_test_data_loader(10)
    ConfusionMatrix(cnn,loader)

if __name__=='__main__':
    CombinationTest()
    #plot_confusion_matrix()