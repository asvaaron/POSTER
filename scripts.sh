
# Path Location Google Collab
cd /content/drive/MyDrive/TEC/Tesis-TEC/CODE/POSTER_Aaron


# Google Collab Location
# Training multiple models in parallel with nohup

# Train Dataset

# Train small model without weights
nohup python train.py --gpu 0 --batch_size 100 --dataset rafdb --modeltype small --epochs 250 > training_small_no_weights.log 2>&1 &

# Train base model without weights
nohup python train.py --gpu 0 --batch_size 100 --dataset rafdb --modeltype base --epochs 250 > training_base_no_weights.log 2>&1 &

# Train large model without weights
nohup python train.py --gpu 0 --batch_size 120 --dataset rafdb --modeltype large --epochs 250 > training_large_no_weights.log 2>&1 &

# Train base model with pretrained weights
nohup python train.py --gpu 0 --batch_size 100 --dataset rafdb --checkpoint /models/weights.ph --modeltype base --epochs 250 > training_base_weights.log 2>&1 &

# Wait for all processes to complete
wait


# LOCAL
nohup python train.py --gpu 0 --batch_size 95 --dataset rafdb --epochs 250 > training.log &

nohup python train.py --gpu 0,1 --batch_size 150 --dataset rafdb --epochs 250 --checkpoint epoch10_acc0.9169.pth > training.log &

nohup python train.py --gpu 0,1 --batch_size 150 --dataset rafdb --modeltype base --epochs 250 --checkpoint checkpoint/epoch81_acc0.9234.pth >> training.log &


## Test Dataset


# Test small model
python test.py --gpu 0,1 --batch_size 100 --dataset rafdb --modeltype small --checkpoint checkpoint/training_small_no_changes/epoch187_acc0.9237.pth -p

python test.py --gpu 0 --batch_size 100 --dataset rafdb --modeltype small --checkpoint checkpoint/training_small_no_changes/epoch187_acc0.9237.pth

# Test base model

python test.py --gpu 0,1 --batch_size 100 --dataset rafdb --modeltype base --checkpoint checkpoint/training_base_no_changes/epoch97_acc0.9244.pth -p

python test.py --gpu 0 --batch_size 100 --dataset rafdb --modeltype base --checkpoint checkpoint/training_base_no_changes/epoch97_acc0.9244.pth

# Test Large model

python test.py --gpu 0 --batch_size 100 --dataset rafdb --modeltype large --checkpoint checkpoint/training_large_no_changes/epoch202_acc0.9234.pth -p
