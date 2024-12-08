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
def var_avg(filepaths,metrics=None):

    for _,filepath in enumerate(filepaths):
        _,y = read_txtdata(filepath,metrics=(metrics+'.txt'))

        y_10epoch = y[len(y)-10:len(y)+1]
        y_mean = np.mean(y_10epoch)
        y_std = np.std(y_10epoch)
        print('***{}***最后10个epoch：'.format(metrics))
        print('均值：{}，标准差：{}'.format(y_mean,y_std))


var_avg(args.path,metrics='val_prec1')
var_avg(args.path,metrics='val_prec5')