# Training

nohup python train.py --gpu 0,1 --batch_size 120 --dataset rafdb --modeltype large --epochs 250 > training.log &
nohup python train.py --gpu 0,1 --batch_size 120 --dataset rafdb --modeltype base --epochs 250 > training.log &
nohup python train.py --gpu 0,1 --batch_size 120 --dataset rafdb --modeltype small --epochs 250 > training.log &

# Testing
python test.py --gpu 0,1 --batch_size 120 --dataset rafdb --modeltype base --head_type simple --checkpoint checkpoint/local_base/epoch162_acc0.9254.pth -p
python test.py --gpu 0,1 --batch_size 120 --dataset rafdb --modeltype large --head_type simple --checkpoint checkpoint/training_large_no_changes/epoch205_acc0.9241.pth -p

python test.py --gpu 0,1 --batch_size 120 --dataset rafdb --modeltype base --head_type enhanced_v1 --checkpoint checkpoint/local_base_enhanced_v1/epoch219_acc0.9247.pth -p


python test.py --gpu 0,1 --batch_size 120 --dataset rafdb --modeltype base --head_type enhanced_v2 --checkpoint checkpoint/local_base_enhanced_v2/epoch58_acc0.9221.pth -p

python test.py --gpu 0,1 --batch_size 120 --dataset rafdb --modeltype large --head_type enhanced_v3 --checkpoint checkpoint/local_base_enhanced_v3/epoch227_acc0.9254.pth -p