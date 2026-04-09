# Training

nohup python train.py --gpu 0,1 --batch_size 120 --dataset rafdb --modeltype large --head_type simple --epochs 250 > training.log &
nohup python train.py --gpu 0,1 --batch_size 120 --dataset rafdb --modeltype base --head_type simple --epochs 250 > training.log &
nohup python train.py --gpu 0,1 --batch_size 120 --dataset rafdb --modeltype small  --head_type simple --epochs 250 > training.log &



nohup python train.py --gpu 0,1 --batch_size 120 --dataset rafdb --modeltype large --head_type enhanced_v1 --epochs 250 > training.log &
nohup python train.py --gpu 0,1 --batch_size 120 --dataset rafdb --modeltype base --head_type enhanced_v1 --epochs 250 > training.log &
nohup python train.py --gpu 0,1 --batch_size 120 --dataset rafdb --modeltype small --head_type enhanced_v1 --epochs 250 > training.log &


# Testing
python test.py --gpu 0,1 --batch_size 120 --dataset rafdb --modeltype base --head_type simple --checkpoint checkpoint/local_base_simple/epoch222_acc0.9276.pth -p
python test.py --gpu 0,1 --batch_size 120 --dataset rafdb --modeltype small --head_type simple --checkpoint checkpoint/local_small_simple/epoch179_acc0.9237.pth -p
python test.py --gpu 0,1 --batch_size 120 --dataset rafdb --modeltype large --head_type simple --checkpoint checkpoint/local_large_simple/epoch200_acc0.925.pth -p


python test.py --gpu 0,1 --batch_size 120 --dataset rafdb --modeltype base --head_type simple --checkpoint checkpoint/training_base_no_changes/epoch97_acc0.9244.pth -p
python test.py --gpu 0,1 --batch_size 120 --dataset rafdb --modeltype large --head_type simple --checkpoint checkpoint/training_large_no_changes/epoch205_acc0.9241.pth -p
python test.py --gpu 0,1 --batch_size 120 --dataset rafdb --modeltype small --head_type simple --checkpoint checkpoint/training_small_no_changes/epoch187_acc0.9237.pth -p

python test.py --gpu 0,1 --batch_size 120 --dataset rafdb --modeltype base --head_type enhanced_v1 --checkpoint checkpoint/local_base_enhanced_v1/epoch219_acc0.9247.pth -p
python test.py --gpu 0,1 --batch_size 120 --dataset rafdb --modeltype large --head_type enhanced_v1 --checkpoint checkpoint/local_large_enchanced_v1/epoch187_acc0.9254.pth -p
python test.py --gpu 0,1 --batch_size 120 --dataset rafdb --modeltype small --head_type enhanced_v1 --checkpoint checkpoint/local_small_enhanced_v1/epoch191_acc0.9234.pth -p

# MC hein tests
python mcnemar_test.py \
  --gpu 0,1 \
  --dataset rafdb \
  --checkpoints \
    checkpoint/training_small_no_changes/epoch187_acc0.9237.pth \
    checkpoint/local_small_enhanced_v1/epoch191_acc0.9234.pth\
    checkpoint/training_base_no_changes/epoch97_acc0.9244.pth \
    checkpoint/local_base_enhanced_v1/epoch219_acc0.9247.pth \
    checkpoint/training_large_no_changes/epoch205_acc0.9241.pth \
    checkpoint/local_large_enchanced_v1/epoch187_acc0.9254.pth \
  --model_names small_baseline small_modified base_baseline base_modified large_baseline large_modified \
  --model_types small small base base large large \
  --head_types simple enhanced_v1 simple enhanced_v1 simple enhanced_v1



python test.py --gpu 0,1 --batch_size 120 --dataset rafdb --modeltype base --head_type enhanced_v2 --checkpoint checkpoint/local_base_enhanced_v2/epoch58_acc0.9221.pth -p

python test.py --gpu 0,1 --batch_size 120 --dataset rafdb --modeltype large --head_type enhanced_v3 --checkpoint checkpoint/local_base_enhanced_v3/epoch227_acc0.9254.pth -p

python test.py --gpu 0,1 --batch_size 120 --dataset rafdb --modeltype large --head_type enhanced_v7 --checkpoint checkpoint/local_large_enhanced_v7/epoch71_acc0.9244.pth -p