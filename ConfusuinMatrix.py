import itertools
import numpy as numpy
import matplotlib.pyplot as plt
import torch
def plot_confusion_matrix(cm,classes,normalize=False,title='Confusion matrix',cmp=plt.cm.Blues):
    if normalize:
        cm=cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
    print(cm)
    plt.imshow(cm,interpolation='nearest',cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks=np.arange(len(classes))
    plt.xticks(tick_marks,classes,rotation=45)
    plt.yticks(tick_marks,classes)

    fmt='.2f' if normalize else 'd'
    thresh=cm.max()/2.
    for i,j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
        plt.text(j,i,format(cm[i,j],fmt),
        horizontalalignment="center",
        color="white" if cm[i,j]>thresh else "black"
        )
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def ConfusionMatrix(network,loader):
    all_preds=torch.tensor([])
    all_labels=torch.tensor([])
    network.eval()
    for batch in loader:
        images,labels=batch
        preds=network(images)
        all_preds=torch.cat((all_preds,preds),dim=0)
        all_labels=torch.cat((all_labels,labels),dim=0)
    stacked=torch.stack((all_labels,all_preds.argmax(dim=1)),dim=1)
    cm=torch.zeros(10,10,dtype=torch.int64)
    for p in stacked:
        j,k=p.tolist()
        cm[j,k]=cm[j,k]+1
    classes=['0','1','2','3','4','5','6','7','8','9']
    plot_confusion_matrix(cm,classes,normalize=False,title='Confusion matrix',cmp=plt.cm.Blues)
