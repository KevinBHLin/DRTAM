# -*- coding: UTF-8 -*-
import argparse
import os
import random
import shutil
import time
import warnings
import sys

# print(sys.argv)

# sys.argv  = sys.argv + ['--arch', 'resnet50', '--cfg', 'cfg_example.py', '--action', 'drtamv12',
#                         '/public/linzengrong/mmclassification/data/imagenet']
#os.system('python main.py --arch resnet50 --cfg cfg_example.py --action drtamv12 /public/linzengrong/mmclassification/data/imagenet')

#import sys
# # print(sys.argv)
#os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'
#sys.argv  = sys.argv + ['--arch', 'mobilenet_v2', '--cfg', '/public/linzengrong/AttenNet-master/cfg_list_example.py', 
#                         '/public/linzengrong/mmclassification/data/imagenet']
#os.system('python main.py --arch resnet50 --cfg cfg_example.py --action drtamv12 /public/linzengrong/mmclassification/data/imagenet')

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models
from mmcv import Config, DictAction



model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--cfg', metavar='config', default='', type=str,
                    help='filename of configuration  (default: None)')
parser.add_argument('--cfg2', metavar='config', default='', type=str,
                    help='filename of configuration  (default: None)')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int, nargs='+',
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--ksize', default=None, type=list,
                    help='Manually select the eca module kernel size')
parser.add_argument('--action', default='', type=str,
                    help='other information.')
parser.add_argument('--action2', default='', type=str,
                    help='other information.')
                    

best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    args.distributed = args.world_size > 1

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)

##########################################################################################

    cfgList = None
    if args.cfg is not None:
        cfgList = read_config(args.cfg)
        #cfg_name = os.path.basename(args.cfg).split('.')[0]
    
    directory_list = []
    model_list = []
    criterion_list = []
    optimizer_list = []
    name_list = []

##########################################################################################
    for cfg in cfgList:
        plugins = cfg['plugins']
        cfg_name = cfg['name']
        resume = None if not hasattr(cfg,'resume') else cfg['resume']

        # create model
        if args.pretrained:
            print("=> using pre-trained model '{}'".format(args.arch))
            model = models.__dict__[args.arch](k_size=args.ksize, pretrained=True, plugins=plugins)
        else:
            print("=> creating model '{}'".format(args.arch))
            if args.ksize == None:
                model = models.__dict__[args.arch](plugins=plugins)
            else:
                model = models.__dict__[args.arch](k_size=args.ksize, plugins=plugins)

        if args.gpu is not None:
            model = model.cuda(args.gpu)
        elif args.distributed:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)

        else:
            if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
                model.features = torch.nn.DataParallel(model.features)
                model.cuda()
            else:
                model = torch.nn.DataParallel(model).cuda()

        print(model)

        # get the number of models parameters
        print('Number of models parameters: {}'.format(
            sum([p.data.nelement() for p in model.parameters()])))

        # define loss function (criterion) and optimizer
        criterion = nn.CrossEntropyLoss().cuda(args.gpu)

        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)

        # optionally resume from a checkpoint
        if resume is not None:
            if os.path.isfile(resume):
                print("=> loading checkpoint '{}'".format(resume))
                checkpoint = torch.load(resume)
                args.start_epoch = checkpoint['epoch']
                best_prec1 = checkpoint['best_prec1']
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {})"
                    .format(resume, checkpoint['epoch']))
                del checkpoint
            else:
                print("=> no checkpoint found at '{}'".format(resume))

        directory = "runs/%s/"%(args.arch + '_' + cfg_name + '_' + args.action)
        if not os.path.exists(directory):
            os.makedirs(directory)

        directory_list.append(directory)
        model_list.append(model)
        criterion_list.append(criterion)
        optimizer_list.append(optimizer)
        name_list.append(cfg_name)

##########################################################################################

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        m = time.time()
        _, _ =validate(val_loader, model_list, criterion_list)
        n = time.time()
        print((n-m)/3600)
        return

    Loss_plot_list = [{} for i in range(len(name_list))]
    train_prec1_plot_list = [{} for i in range(len(name_list))]
    train_prec5_plot_list = [{} for i in range(len(name_list))]
    val_prec1_plot_list = [{} for i in range(len(name_list))]
    val_prec5_plot_list = [{} for i in range(len(name_list))]
    rest_time =  float("inf") 

    for epoch in range(args.start_epoch, args.epochs):
        start_time = time.time()
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer_list, epoch)

        # train for one epoch
        # train(train_loader, model, criterion, optimizer, epoch)
        loss_temp_list, train_prec1_temp_list, train_prec5_temp_list = train(train_loader, model_list, criterion_list, optimizer_list, epoch, rest_time, args.epochs)
        prec1_list, prec5_list = validate(val_loader, model_list, criterion_list)

        ##########################################################################################
        for i in range(len(name_list)):
            Loss_plot = Loss_plot_list[i]
            train_prec1_plot = train_prec1_plot_list[i]
            train_prec5_plot = train_prec5_plot_list[i]
            val_prec1_plot = val_prec1_plot_list[i]
            val_prec5_plot = val_prec5_plot_list[i]
            model = model_list[i]
            optimizer = optimizer_list[i]
            cfg_name = name_list[i]
            directory = directory_list[i]

            Loss_plot[epoch] = loss_temp_list[i]
            train_prec1_plot[epoch] = train_prec1_temp_list[i]
            train_prec5_plot[epoch] = train_prec5_temp_list[i]

            # evaluate on validation set
            # prec1 = validate(val_loader, model, criterion)
            val_prec1_plot[epoch] = prec1_list[i]
            val_prec5_plot[epoch] = prec5_list[i]

            # remember best prec@1 and save checkpoint
            is_best = prec1_list[i] > best_prec1
            best_prec1 = max(prec1_list[i], best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'cfg' : cfg_name,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer' : optimizer.state_dict(),
            }, is_best)
            
            # 将Loss,train_prec1,train_prec5,val_prec1,val_prec5用.txt的文件存起来
            data_save(directory + 'Loss_plot.txt', Loss_plot)
            data_save(directory + 'train_prec1.txt', train_prec1_plot)
            data_save(directory + 'train_prec5.txt', train_prec5_plot)
            data_save(directory + 'val_prec1.txt', val_prec1_plot)
            data_save(directory + 'val_prec5.txt', val_prec5_plot)
        ##########################################################################################

        end_time = time.time()
        time_value = (end_time - start_time) / 3600
        rest_time = time_value*(args.epochs-args.start_epoch-1)
        print("-" * 80)
        print("cost time: ", time_value, ", rest time: ", rest_time)
        print("-" * 80)


def train(train_loader, models, criterions, optimizers, epoch, rest_time, end_epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses_list = []
    top1_list = []
    top5_list = []
    losses_avg_list = []
    top1_avg_list = []
    top5_avg_list = []
    for i in range(len(models)):
        losses_list.append(AverageMeter())
        top1_list.append(AverageMeter())
        top5_list.append(AverageMeter())
    # switch to train mode
    for model in models:
        model.train()

    batch_num = len(train_loader)
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        for point, (model, criterion, optimizer) in enumerate(zip(models,criterions,optimizers)):
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses_list[point].update(loss.item(), input.size(0))
            top1_list[point].update(prec1[0], input.size(0))
            top5_list[point].update(prec5[0], input.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        rest_time = (batch_time.avg*(batch_num-i)+(batch_time.avg*batch_num)*(end_epoch-epoch-1))/3600

        if i % args.print_freq == 0:
            for point in range(len(losses_list)):
                print('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})({rest_time:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Model{model_label: d}\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time, rest_time=rest_time,
                    data_time=data_time, model_label=point,
                    loss=losses_list[point], top1=top1_list[point], top5=top5_list[point]))
        
    for point in range(len(losses_list)):
        losses_avg_list.append(losses_list[point].avg)
        top1_avg_list.append(top1_list[point].avg)
        top5_avg_list.append(top5_list[point].avg)
        
    return losses_avg_list, top1_avg_list, top5_avg_list


def read_config(filename):
    modelList = None
    cfg = Config.fromfile(filename)
    modelList = cfg.modelList
    # for i, plugin in enumerate(plugins):
    #     name = name + plugin.pop('name','m'+str(i))
    #     tmp = plugin['cfg'].copy()
    #     del tmp['type']
    #     for v in tmp.values():
    #         name = name + '_' + str(v)
    #     tmp_str = ''
    #     for s in plugin.pop('stages',[]):
    #         tmp_str = tmp_str + '1' if s else tmp_str + '0'
    #     name = name + '_' + tmp_str
    #     name = name + '_p' + plugin.pop('position','')[-1]
    return modelList

def validate(val_loader, models, criterions):
    batch_time = AverageMeter()

    losses_list = []
    top1_list = []
    top5_list = []

    top1_avg_list = []
    top5_avg_list = []

    for i in range(len(models)):
        losses_list.append(AverageMeter())
        top1_list.append(AverageMeter())
        top5_list.append(AverageMeter())

    # switch to evaluate mode
    for model in models:
        model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            for point, (model, criterion) in enumerate(zip(models,criterions)):
                # compute output
                output = model(input)
                loss = criterion(output, target)

                # measure accuracy and record loss
                prec1, prec5 = accuracy(output, target, topk=(1, 5))
                losses_list[point].update(loss.item(), input.size(0))
                top1_list[point].update(prec1[0], input.size(0))
                top5_list[point].update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                for point in range(len(losses_list)):
                    print('Test: [{0}/{1}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Model{model_label: d}\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                        'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        i, len(val_loader), batch_time=batch_time, model_label=point, loss=losses_list[point],
                        top1=top1_list[point], top5=top5_list[point]))
        
        for point in range(len(losses_list)):
            print(' * Model{model_label: d} Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
                .format(model_label=point, top1=top1_list[point], top5=top5_list[point]))
            top1_avg_list.append(top1_list[point].avg)
            top5_avg_list.append(top5_list[point].avg)
    return top1_avg_list, top5_avg_list


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    directory = "runs/%s/"%(args.arch + '_' + state['cfg'] + '_' + args.action)
    
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, directory + 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer_list, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for optimizer in optimizer_list:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def data_save(root, file):
    if not os.path.exists(root):
        os.mknod(root)
    file_temp = open(root, 'r')
    lines = file_temp.readlines()
    if not lines:
        epoch = -1
    else:
        epoch = lines[-1][:lines[-1].index(' ')]
    epoch = int(epoch)
    file_temp.close()
    file_temp = open(root, 'a')
    for line in file:
        if line > epoch:
            file_temp.write(str(line) + " " + str(file[line]) + '\n')
    file_temp.close()


if __name__ == '__main__':
    main()
