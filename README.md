# Plastic_Segmentation
Training and inference of a Deep Learning Model for segmenting plastic particle captured with SEM

# Prepare Plastic Dataset
Download images and annotations at: https://drive.google.com/drive/folders/19OfFWVn8J_V-zTI5V_ZcGC0xpxUNK7MV?usp=drive_link

The images are annotated plastic particles and their corresponding GT masks coupled with cells data and thheir GT masks.

Place the images under the folder "plastic_data/"

# Training the Deep Model
python train.py --dataset plastic --model "fusion/dilation/unet/tunet/tunet2/DVLab3" --train_data plastic_data/

# Inference
Modify l100-101 in inference2.py to choose which video to process

-- ptype = "010_Blinded1/4_BottomRight" #  Kaolinite - Cellulose - HDPE - HDPEre - PETE - PETEre - PVC - PVCre - Algae - 010_Blinded1/1_TopLeft/2_TopRight/3_BottomLeft/4_BottomRight

-- output_dir = ""

python inference2.py --dataset plastic --model "fusion/dilation/unet/tunet/tunet2/DVLab3"

# Finetuned models
You can download finetuned models at: https://drive.google.com/drive/folders/1AsvD-vzg2JMjTR81WSGnwd0q_fkDJHQM?usp=drive_link

