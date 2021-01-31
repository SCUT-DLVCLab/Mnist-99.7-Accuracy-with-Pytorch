import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F

from RunBuilder import RunBuilder
from cnnModel import CNN
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict
from Mydataset import get_train_data_loader
from Mydataset import get_test_data_loader
#saved path
SAVED_PATH="E:\LearningStuff\DLcode\Pytorch\Mnist"

#hyperParameters
num_epochs=30
#TestParameters
params=OrderedDict(
learning_rate=[0.001]
,batch_size=[256]
,device=['cuda']
)

def get_num_correct(preds,labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

def adjust_learning_rate(optimizer, epoch,learning_rate):
    """Sets the learning rate to be 0.0001 after 20 epochs"""
    if epoch>=20:
        learning_rate=0.0001
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate

def main():

    print("Initalizing Network")
    for run in RunBuilder.get_runs(params):
            comment=f'-{run}'
    device=torch.device(run.device)
    cnn=CNN()
    cnn=cnn.to(device)
    #cnn.load_state_dict(torch.load("E:\LearningStuff\DLcode\Pytorch\Mnist\CNN3MODEL9967.pkl"))
    #extending_loader=get_expanding_data_loader([8],length=2000,batch_size=run.batch_size)
    train_loader=get_train_data_loader(run.batch_size)
    test_loader=get_test_data_loader(run.batch_size)
    optimizer=optim.Adam(cnn.parameters(),lr=run.learning_rate)

    ''' Initializing tensorboard '''
    tb=SummaryWriter(comment=comment,flush_secs=1)
    images,labels=next(iter(train_loader))
    grid=torchvision.utils.make_grid(images)

    '''begin to train'''
    for epoch in range(num_epochs):
        total_loss=0
        total_correct=0
        adjust_learning_rate(optimizer, epoch,run.learning_rate)
        for batch in train_loader:
            images=batch[0].to(device)
            labels=batch[1].to(device)
            preds=cnn(images)
            loss=F.cross_entropy(preds,labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss+=loss.item()
            total_correct+=get_num_correct(preds,labels)
        tb.add_scalar('Loss',total_loss,epoch)
        tb.add_scalar('Number Correct',total_correct,epoch)
        tb.add_scalar('Accuracy',total_correct/60000,epoch)
        path=SAVED_PATH+"\MODEL"+str(epoch)+".pkl"
        torch.save(cnn.state_dict(),path)
        print("epoch",epoch,"loss",total_loss,"Accuracy",total_correct/60000)
    
    cnn.to('cpu').eval()
    for epoch in range(num_epochs):
        cnn.load_state_dict(torch.load(SAVED_PATH+"\MODEL"+str(epoch)+".pkl"))
        total_correct=0
        for batchs in test_loader:
            images,labels=batchs
            preds=cnn(images)
            total_correct+=get_num_correct(preds,labels)
        print("The Accuracy of the ",epoch,"model on test-set is:",total_correct/60000)
        tb.add_scalar('Accuracy on testset',total_correct/10000,epoch)
    tb.close()

if __name__=='__main__':
    main()