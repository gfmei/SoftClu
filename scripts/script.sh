# Traning CrossPoint for classification
python main.py --model dgcnn --epochs 100 --lr 0.001 --exp_name dgcnn_dis_64 --num_clus 64 --batch_size 32 --print_freq 200 --k 20
# Training CrossPoint for part-segmentation
python main.py --model dgcnn_seg --epochs 100 --lr 0.001 --exp_name dgcnn_seg_64 --batch_size 20 --print_freq 200 --k 20
# Fine-tuning for part-segmentation
python train_partseg.py --exp_name dgcnn_seg_64 --pretrained_path best_model.pth --batch_size 12 --k 40 --test_batch_size 8 --epochs 300


python main.py --model pointnet --epochs 100 --lr 0.001 --exp_name pnet_cls_M64x --num_clus 64 --batch_size 24 --print_freq 200 --k 20 --subset10 True --ablation xyz  --dataset modelnet

python main.py --model pointnet --epochs 100 --lr 0.001 --exp_name pnet_cls_M64f --num_clus 64 --batch_size 24 --print_freq 200 --k 20 --subset10 True --ablation fea  --dataset modelnet


python main.py --model dgcnn --epochs 100 --lr 0.001 --exp_name dgcnn_dis_64 --num_clus 64 --batch_size 32 --print_freq 200 --k 64 --ablation xyz --c_type  dis --dataset modelnet



python train_partseg.py --exp_name dgcnn_rand --batch_size 12 --k 40 --test_batch_size 8 --epochs 300