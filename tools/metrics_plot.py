import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
parser = argparse.ArgumentParser(description='Plot')
parser.add_argument('--path', default=None, type=str,nargs='+',
                    help='filepath of valfile')                   

parser.add_argument('--legend', default=None, type=str,nargs='+',
                    help='path of save')
parser.add_argument('--only_difference', default=False, type=bool,
                    help='only plot difference between train and val ')
global args

savepath = '/home/linzengrong/AttenNet-master/figures/'
args = parser.parse_args()

def read_txtdata(filepath,metrics=None):
    matrix = []
    filepath = filepath + metrics
    f = open(filepath,'r')
    print('***plotting '+filepath+'***')
    for line in f.readlines():
        line = line.replace('tensor(','')
        line = line.replace(', device=\'cuda:0\')','')
        line = line.strip()
        line = line.split(' ')
        matrix.append(line)
    matrix = np.float128(matrix)
    x = matrix[:,0]
    x = x+1
    y = matrix[:,1]
    return x,y

def plot_accuracy(filepaths,title=None,metrics=None,savepath=savepath,flag=False):
    savepath = savepath + '/' + metrics + '.png'
    plt.figure(figsize=(20,8))
    plt.title(title,fontdict={"size": 22})
    plt.grid(True,linestyle='--',alpha=0.5)
    plt.xlabel('Epoch',fontsize=18)
    plt.xlim(1,100)
    plt.ylabel('Accuracy(%)',fontsize=18)
    xtick = [1,30,60,90,100]     
    plt.xticks(xtick)
    for _,filepath in enumerate(filepaths):
        x,y = read_txtdata(filepath,metrics=(metrics+'.txt'))
        if np.max(y)>80:
            ytick = np.arange(10,105,5)
        else:
            ytick = np.arange(20,85,5)
        if flag is not False :
            y_10epoch = y[len(y)-10:len(y)+1]
            y_mean = np.mean(y_10epoch)
            y_std = np.std(y_10epoch)
            print('***{}***最后10个epoch：'.format(metrics))
            print('均值：{}，标准差：{}'.format(y_mean,y_std))
        plt.yticks(ytick)
        plt.plot(x,y)
        plt.savefig(savepath,dpi=300)

    plt.legend(labels,prop={"size": 15})
    plt.savefig(savepath,dpi=300)

def plot_loss(filepaths,title=None,metrics=None,savepath=savepath):
    savepath = savepath + '/' + metrics + '.png'
    plt.figure(figsize=(20,8))
    plt.title(title,fontdict={"size": 22})
    plt.grid(True,linestyle='--',alpha=0.5)
    plt.xlabel('Epoch',fontsize=18)
    plt.xlim(1,100)
    plt.ylabel('Loss',fontsize=18)
    xtick = [1,30,60,90,100]     
    plt.xticks(xtick)   
    for _,filepath in enumerate(filepaths):
        x,y = read_txtdata(filepath,metrics=(metrics+'.txt'))
        plt.plot(x,y)
        plt.savefig(savepath,dpi=300)
    plt.legend(labels,prop={"size": 15})
    plt.savefig(savepath,dpi=300)

def plot_dif_accuracy(filepaths,title=None,metrics=None,savepath=savepath):
    savepath = savepath + '/' + 'val-train difference' + '.png'
    plt.figure(figsize=(20,8))
    plt.title(title,fontdict={"size": 22})
    plt.grid(True,linestyle='--',alpha=0.5)
    plt.xlabel('Epoch',fontsize=18)
    plt.xlim(1,100)
    plt.ylabel('Accuracy(%)',fontsize=18)
    xtick = [1,30,60,61,62,63,90,100]     
    plt.xticks(xtick)
    for _,filepath in enumerate(filepaths):
        i = 0
        for _,metric in enumerate(metrics):
            x,z = read_txtdata(filepath,metrics=(metric+'.txt'))
            if i==0: 
                y = np.zeros((len(z),2),dtype=float)
            y[:,i] = z
            i = i+1
        plt.plot(x,y[:,0]-y[:,1])
        plt.savefig(savepath,dpi=300)

    plt.legend(labels,prop={"size": 15})
    plt.savefig(savepath,dpi=300)



labels=[]
for _,label in enumerate(args.legend):
    labels.append(label)
    savepath = savepath  + label + '_' 
if not os.path.exists(savepath):  
    os.makedirs(savepath)
if args.only_difference is False:
    plot_accuracy(args.path,title='Val-1-Accuracy Curve',metrics='val_prec1',savepath=savepath,flag=True)
    plot_accuracy(args.path,title='Val-5-Accuracy Curve',metrics='val_prec5',savepath=savepath,flag=True)
    plot_accuracy(args.path,title='Train-1-Accuracy Curve',metrics='train_prec1',savepath=savepath,flag=False)
    plot_accuracy(args.path,title='Train-5-Accuracy Curve',metrics='train_prec5',savepath=savepath,flag=False)
    plot_dif_accuracy(args.path,title='val-train Accuracy difference',metrics=['val_prec1','train_prec1'],savepath=savepath)
    plot_loss(args.path,title='Val-Loss Curve',metrics='Loss_plot',savepath=savepath)
else :
    plot_dif_accuracy(args.path,title='val-train Accuracy difference',metrics=['val_prec1','train_prec1'],savepath=savepath)

