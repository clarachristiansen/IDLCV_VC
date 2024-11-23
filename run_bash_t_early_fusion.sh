#!/bin/sh

#BSUB -q c02516

#BSUB -gpu "num=1:mode=exclusive_process"

#BSUB -J testjob

#BSUB -n 4

#BSUB -R "span[hosts=1]"

#BSUB -R "rusage[mem=20GB]"

#BSUB -W 12:00

#BSUB -o early_fusion_resnet18%J.out
#BSUB -e early_fusion_resnet18%e.err

source /zhome/88/7/117159/venv/bin/activate

python /zhome/88/7/117159/Courses/IDLCV_VC/early_fusion.py
