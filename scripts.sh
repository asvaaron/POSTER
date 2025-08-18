
# Path Location Google Collab
cd /content/drive/MyDrive/TEC/Tesis-TEC/CODE/POSTER_Aaron


# Google Collab Location
# Training multiple models in parallel with nohup

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

nohup python train.py --gpu 0,1 --batch_size 150 --dataset rafdb --epochs 250 > training.log &
