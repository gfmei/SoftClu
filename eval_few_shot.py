import argparse
import os
import random

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch import nn
from tqdm import tqdm

from data.modelnet import load_modelnet_data, load_scanobjectnn
from models.encoder import DGCNN, PointNet
from models.model import ClusterNet

parser = argparse.ArgumentParser(description='Point Cloud Recognition')
parser.add_argument('--root', type=str, default='/home/gmei/Data/data', help="dataset path")
parser.add_argument('--num_points', type=int, default=1024,
                    help='num of points to use')
parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                    help='Dimension of embeddings')
parser.add_argument('--k', type=int, default=20, metavar='N',
                    help='Num of nearest neighbors to use')
parser.add_argument('--num_clus', type=int, default=64, metavar='N',
                    help='Num of nearest neighbors to use')
parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
                    choices=['modelnet40', 'scanobjectnn'],
                    help='Dataset to evaluate')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='dropout rate')
parser.add_argument('--n_runs', type=int, default=10,
                    help='Num of few-shot runs')
parser.add_argument('--k_way', type=int, default=5,
                    help='Num of classes in few-shot')
parser.add_argument('--m_shot', type=int, default=20,
                    help='Num of samples in one class')
parser.add_argument('--n_query', type=int, default=20,
                    help='Num of query samples in one class')
parser.add_argument('--model_path', type=str, default='pointnet_ot_64MF', metavar='N',
                    help='Pretrained model path')
args = parser.parse_args()

device = torch.device("cuda")

# Try to load models
# net = DGCNN(args.emb_dims, args.k, args.dropout, -1).to(device)
net = PointNet(args.emb_dims, is_normal=False, feature_transform=True, feat_type='global').to(device)
# model_path = os.path.join(f'checkpoints/{args.exp_name}/models/', 'best_model.pth')
model_path = os.path.join(f'checkpoints/{args.model_path}/models/', 'best_model.pth')
model = ClusterNet(net, dim=args.emb_dims, num_clus=args.num_clus, ablation='all', c_type='ot')
try:
    model.load_state_dict(torch.load(model_path))
except Exception as e:
    print('No existing model, starting training from scratch {}'.format(e))
model.load_state_dict(torch.load(model_path))
model.inv_head = nn.Identity()
print("Model Loaded !!")

if args.dataset == 'modelnet40':
    # ModelNet40 - Few Shot Learning
    data_train, label_train = load_modelnet_data(args.root, 'train')
    data_test, label_test = load_modelnet_data(args.root, 'test')
    n_cls = 40
elif args.dataset == 'scanobjectnn':
    # ScanObjectNN - Few Shot Learning
    data_train, label_train = load_scanobjectnn(args.root, 'train')
    data_test, label_test = load_scanobjectnn(args.root, 'test')
    n_cls = 15
else:
    raise NotImplementedError

label_idx = {}
for key in range(n_cls):
    label_idx[key] = []
    for i, label in enumerate(label_train):
        # if label[0] == key:
        if label == key:
            label_idx[key].append(i)

acc = []
for run in tqdm(range(args.n_runs)):
    k = args.k_way
    m = args.m_shot
    n_q = args.n_query

    k_way = random.sample(range(n_cls), k)

    data_support = []
    label_support = []
    data_query = []
    label_query = []
    for i, class_id in enumerate(k_way):
        support_id = random.sample(label_idx[class_id], m)
        query_id = random.sample(list(set(label_idx[class_id]) - set(support_id)), n_q)

        pc_support_id = data_train[support_id]
        pc_query_id = data_train[query_id]
        data_support.append(pc_support_id)
        label_support.append(i * np.ones(m))
        data_query.append(pc_query_id)
        label_query.append(i * np.ones(n_q))

    data_support = np.concatenate(data_support)
    label_support = np.concatenate(label_support)
    data_query = np.concatenate(data_query)
    label_query = np.concatenate(label_query)

    feats_train = []
    labels_train = []
    model = model.eval()

    for i in range(k * m):
        data = torch.from_numpy(np.expand_dims(data_support[i], axis=0))
        label = int(label_support[i])
        data = data.permute(0, 2, 1).to(device)
        data = torch.cat((data, data))
        with torch.no_grad():
            feat = model.backbone(data)[0][0, :]
        feat = feat.detach().cpu().numpy().tolist()
        feats_train.append(feat)
        labels_train.append(label)
    feats_train = np.array(feats_train)
    labels_train = np.array(labels_train)
    feats_test = []
    labels_test = []

    for i in range(k * n_q):
        data = torch.from_numpy(np.expand_dims(data_query[i], axis=0))
        label = int(label_query[i])
        data = data.permute(0, 2, 1).to(device)
        data = torch.cat((data, data))
        with torch.no_grad():
            feat = model.backbone(data)[0][0, :]
        feat = feat.detach().cpu().numpy().tolist()
        feats_test.append(feat)
        labels_test.append(label)

    feats_test = np.array(feats_test)
    labels_test = np.array(labels_test)

    # scaler = MinMaxScaler()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(feats_train)
    model_tl = SVC(kernel='linear')
    model_tl.fit(scaled, labels_train)
    # model_tl.fit(feats_train, labels_train)

    test_scaled = scaler.transform(feats_test)

    # accuracy = model_tl.score(feats_test, labels_test) * 100
    accuracy = model_tl.score(test_scaled, labels_test) * 100
    acc.append(accuracy)
print('{} +/- {}'.format(np.mean(acc), np.std(acc)))
