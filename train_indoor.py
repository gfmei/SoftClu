import argparse
import os

import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader

from data.S3DIS import S3DISDatasetHDF5
from data.modelnet import ModelNet40
from lib.lib_utils import AverageMeter, IOStream
from models.encoder import PointNet, DGCNNSemSeg
from models.model import ClusterNet


def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)
    if not os.path.exists('checkpoints/' + args.exp_name + '/' + 'models'):
        os.makedirs('checkpoints/' + args.exp_name + '/' + 'models')


def train(args, io):
    if args.dataset == 's3dis':
        train_loader = DataLoader(S3DISDatasetHDF5(root=args.root, split='train', test_area=8))
    else:
        train_loader = DataLoader(ModelNet40(args.root, num_points=args.num_points, transform=False, train=True,
                                             normalize=True, subset10=args.subset10), num_workers=args.workers,
                                  batch_size=args.batch_size, shuffle=True, drop_last=True)
    # device = torch.device("cuda" if args.cuda else "cpu")
    # Try to load models
    if args.model == 'dgcnn_sem':
        net = DGCNNSemSeg(args.emb_dims, args.k, num_class=40, num_channel=9,
                          dropout=args.dropout, pretrain=True).cuda()
    elif args.model == 'pointnet':
        net = PointNet(args.emb_dims, is_normal=False, feature_transform=True, feat_type='global').cuda()
    else:
        raise Exception("Not implemented")
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    point_model = ClusterNet(net, dim=args.emb_dims, num_clus=args.num_clus, ablation=args.ablation,
                             c_type=args.c_type)
    model_path = os.path.join(f'checkpoints/{args.exp_name}/models/', 'best_model.pth')
    if args.resume:
        try:
            point_model.load_state_dict(torch.load(model_path))
        except Exception as e:
            io.cprint('No existing model, starting training from scratch {}'.format(e))
    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(point_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-6)
    else:
        print("Use Adam")
        opt = torch.optim.AdamW(point_model.parameters(), lr=args.lr, betas=(0.9, 0.999),
                                eps=1e-08, weight_decay=args.decay_rate)

    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=0, last_epoch=-1)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.7)
    best_loss = 1e4
    for epoch in range(args.start_epoch, args.epochs):
        ####################
        # Train
        ####################
        train_losses = AverageMeter()
        point_model.train()
        # wandb_log = {}
        print(f'Start training epoch: ({epoch}/{args.epochs})')
        for i, data in enumerate(train_loader):
            data = data[0].cuda()
            batch_size = data.size()[0]
            opt.zero_grad()
            data = data.transpose(2, 1).contiguous()
            loss_rq, trans_loss = point_model(data)
            total_loss = loss_rq + trans_loss
            total_loss.backward()
            opt.step()
            train_losses.update(total_loss.item(), batch_size)
            if i % args.print_freq == 0:
                print('Epoch (%d), Batch(%d/%d), loss: %.6f' % (
                    epoch, i, len(train_loader), train_losses.avg))
        lr_scheduler.step()

        outstr = 'Train %d, loss: %.6f' % (epoch, train_losses.avg)
        io.cprint(outstr)

        if train_losses.avg < best_loss:
            best_loss = train_losses.avg
            print('==> Saving Best Model...')
            save_file = model_path
            torch.save(point_model.state_dict(), save_file)
        io.cprint(f"Best Loss: {best_loss}")

        if epoch % args.save_freq == 0:
            print('==> Saving...')
            save_file = os.path.join(f'checkpoints/{args.exp_name}/models/',
                                     'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(point_model.state_dict(), save_file)
        # wandb.log(wandb_log)

    print('==> Saving Last Model...')
    save_file = os.path.join(f'checkpoints/{args.exp_name}/models/',
                             'ckpt_epoch_last.pth')
    torch.save(point_model.state_dict(), save_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    # dataset
    parser.add_argument('--root', type=str,
                        default='/home/gmei/Data/data/indoor3d_sem_seg_hdf5_data', help="dataset path")
    parser.add_argument('--dataset', type=str, default='s3dis', metavar='N',
                        choices=['s3dis', 'modelnet'],
                        help='Dataset to use, [pointnet, dgcnn]')
    parser.add_argument('--subset10', action='store_true', default=False,
                        help='Whether to use ModelNet10 [default: False]')
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_area', type=int, default=5, help='test area, 1-6 [default: 5]')
    parser.add_argument('--num_points', type=int, default=2048,
                        help='num of points to use')
    # model
    parser.add_argument('--model', type=str, default='dgcnn_sem', metavar='N',
                        choices=['dgcnn', 'dgcnn_sem', 'pointnet'],
                        help='Model to use, [pointnet, dgcnn]')
    parser.add_argument('--ablation', type=str, default='all',
                        help='[xyz, all, fea]')
    parser.add_argument('--c_type', type=str, default='ot',
                        help='[ot, ds]')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--num_clus', type=int, default=64, metavar='N',
                        help='Num of clusters to use')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    # Training settings
    parser.add_argument('--use_sgd', action="store_true", help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--decay_rate', type=float, default=1e-4,
                        help='decay rate [default: 1e-4]')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool, default=False,
                        help='evaluate the model')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--exp_name', type=str, default='dgcnn_sem_S64', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of episode to train')
    parser.add_argument('--start_epoch', type=int, default=0, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--resume', action="store_true", help='resume from checkpoint')
    parser.add_argument('--save_freq', type=int, default=50, help='save frequency')
    parser.add_argument('--print_freq', type=int, default=50, help='print frequency')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    args = parser.parse_args()

    _init_()

    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))
    torch.manual_seed(args.seed)

    if not args.eval:
        train(args, io)
