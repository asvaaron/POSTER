
# Path Location Google Collab
cd /content/drive/MyDrive/TEC/Tesis-TEC/POSTER/


# CHANGE name of files
for f in *.jpg
do
    mv -v "${f}" "${f%.*}_aligned.${f##*.}"
done

# Google Collab Location
nohup python train.py --gpu 0 --batch_size 125 --dataset rafdb --epochs 250 > training.log &

# LOCAL
nohup python train.py --gpu 0 --batch_size 95 --dataset rafdb --epochs 250 > training.log &
