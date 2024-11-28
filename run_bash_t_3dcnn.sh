#!/bin/sh

#BSUB -q c02516

#BSUB -gpu "num=1:mode=exclusive_process"

#BSUB -J testjob

#BSUB -n 4

#BSUB -R "span[hosts=1]"

#BSUB -R "rusage[mem=20GB]"

#BSUB -W 12:00

#BSUB -o resnet3d%J.out
#BSUB -e resnet3d%e.err

python 3dcnn.py
