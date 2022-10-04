import argparse
import os

import numpy as np
import torch
import torch.optim as optim
from sklearn.svm import SVC
from torch import nn
from torch.utils.data import DataLoader

from data.modelnet import ModelNet40SVM, mnet_dataloader
from data.shapenet import ShapeNetClr
from lib.lib_utils import AverageMeter, IOStream
from models.baseline import MLP, SimCLR
from models.encoder import DGCNN, DGCNNPartSeg, PointNet, DGCNNSemSeg


def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)
    if not os.path.exists('checkpoints/' + args.exp_name + '/' + 'models'):
        os.makedirs('checkpoints/' + args.exp_name + '/' + 'models')


def train(args, io):
    train_loader = DataLoader(ShapeNetClr(args.root, transform=False, n_points=args.num_points),
                              num_workers=args.workers,
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
    # Try to load models
    if args.model == 'dgcnn':
        net = DGCNN(args.emb_dims, args.k, args.dropout, -1).cuda()
    elif args.model == 'dgcnn_seg':
        net = DGCNNPartSeg(args.emb_dims, args.k, args.dropout).cuda()
    elif args.model == 'dgcnn_sem':
        net = DGCNNSemSeg(args.emb_dims, args.k, num_class=40, num_channel=3,
                          dropout=args.dropout, pretrain=True).cuda()
    elif args.model == 'pointnet':
        net = PointNet(args.emb_dims, is_normal=False, feature_transform=True, feat_type='global').cuda()
    else:
        raise Exception("Not implemented")
    projector = MLP(in_size=args.emb_dims, out_size=args.proj_dim)
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    # net = nn.DataParallel(net)
    # projector = nn.DataParallel(projector)
    point_model = SimCLR(net, projector, tau=args.tau)
    model_path = os.path.join(f'checkpoints/{args.exp_name}/models/', 'best_model.pth')
    if args.resume:
        try:
            point_model.load_state_dict(torch.load(model_path), strict=False)
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
    # Testing dataset
    if args.subset10:
        train_val_loader, test_val_loader = mnet_dataloader(
            args.root, 1024, 128, transform=False, normalize=True,
            xyz_only=True, uniform=False, subset10=args.subset10, num_workers=args.workers)
    else:
        train_val_loader = DataLoader(ModelNet40SVM(args.root, partition='train', num_points=1024),
                                      batch_size=128, shuffle=True, num_workers=args.workers)
        test_val_loader = DataLoader(ModelNet40SVM(args.root, partition='test', num_points=1024),
                                     batch_size=128, shuffle=True, num_workers=args.workers)
    best_acc = 0
    for epoch in range(args.start_epoch, args.epochs):
        ####################
        # Train
        ####################
        train_losses = AverageMeter()
        point_model.train()
        # wandb_log = {}
        print(f'Start training epoch: ({epoch}/{args.epochs})')
        for i, data in enumerate(train_loader):
            x1, x2 = data
            x1 = x1.cuda()
            x1 = x1.transpose(2, 1)
            x2 = x2.cuda()
            x2 = x2.transpose(2, 1)
            batch_size = data[0].size()[0]
            opt.zero_grad()
            total_loss = point_model(x1, x2)
            total_loss.backward()
            opt.step()
            train_losses.update(total_loss.item(), batch_size)
            if i % args.print_freq == 0:
                print('Epoch (%d), Batch(%d/%d), loss: %.6f' % (
                    epoch, i, len(train_loader), train_losses.avg))
        lr_scheduler.step()
        outstr = 'Train %d, loss: %.6f' % (epoch, train_losses.avg)
        io.cprint(outstr)

        feats_train = []
        labels_train = []
        point_model.eval()
        for i, (data, label) in enumerate(train_val_loader):
            labels = list(map(lambda x: x[0], label.numpy().tolist()))
            data = data.permute(0, 2, 1).cuda()
            with torch.no_grad():
                feats = point_model.backbone(data)[0]
            feats = feats.detach().cpu().numpy()
            for feat in feats:
                feats_train.append(feat)
            labels_train += labels

        feats_train = np.array(feats_train)
        labels_train = np.array(labels_train)

        feats_test = []
        labels_test = []

        if (epoch % 2) and (epoch < 100):
            continue
        for i, (data, label) in enumerate(test_val_loader):
            labels = list(map(lambda x: x[0], label.numpy().tolist()))
            data = data.permute(0, 2, 1).cuda()
            with torch.no_grad():
                feats = point_model.backbone(data)[0]
            feats = feats.detach().cpu().numpy()
            for feat in feats:
                feats_test.append(feat)
            labels_test += labels

        feats_test = np.array(feats_test)
        labels_test = np.array(labels_test)
        # model_tl = LinearSVC(C=0.1, kernel='linear')
        model_tl = SVC()
        model_tl.fit(feats_train, labels_train)
        test_accuracy = model_tl.score(feats_test, labels_test)
        io.cprint(f"Linear Accuracy: {test_accuracy}")
        print(f"Linear Accuracy: {test_accuracy}")
        if test_accuracy > best_acc:
            best_acc = test_accuracy
            print('==> Saving Best Model...')
            save_file = model_path
            torch.save(point_model.state_dict(), save_file)
        io.cprint(f"Best Accuracy: {best_acc}")

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
    parser = argparse.ArgumentParser(description='SimCLR')
    # dataset
    parser.add_argument('--root', type=str,
                        default='/home/gmei/Data/data', help="dataset path")
    parser.add_argument('--dataset', type=str, default='shapenet', metavar='N',
                        choices=['shapenet', 'modelnet'],
                        help='Dataset to use, [pointnet, dgcnn]')
    parser.add_argument('--subset10', action='store_true', default=False,
                        help='Whether to use ModelNet10 [default: False]')
    parser.add_argument('--batch_size', type=int, default=12, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--num_points', type=int, default=2048,
                        help='num of points to use')
    # model
    parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                        choices=['dgcnn', 'dgcnn_seg', 'pointnet'],
                        help='Model to use, [pointnet, dgcnn]')
    parser.add_argument('--tau', type=float, default=0.01, help='Temperature [default: 0.25]')
    parser.add_argument('--proj_dim', type=int, default=128, help='Project dimension [default: 128]')
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
    parser.add_argument('--exp_name', type=str, default='dgcnn_ot_64M', metavar='N',
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
