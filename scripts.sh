
# Path Location Google Collab
cd /content/drive/MyDrive/TEC/Tesis-TEC/POSTER/


# Google Collab Location
nohup python train.py --gpu 0 --batch_size 125 --dataset rafdb --epochs 250 > training.log &

# LOCAL
nohup python train.py --gpu 0 --batch_size 95 --dataset rafdb --epochs 250 > training.log &
