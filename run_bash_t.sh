#!/bin/sh

#BSUB -q c02516

#BSUB -gpu "num=1:mode=exclusive_process"

#BSUB -J testjob

#BSUB -n 4

#BSUB -R "span[hosts=1]"

#BSUB -R "rusage[mem=20GB]"

#BSUB -W 12:00

#BSUB -o every_frame_classification_resnet18%J.out
#BSUB -e every_frame_classification_resnet18%e.err

source /zhome/88/7/117159/venv/bin/activate

python /zhome/88/7/117159/Courses/IDLCV_VC/every_frame_classification.py
