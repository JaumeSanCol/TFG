#!/bin/bash
#SBATCH --gpus=1
#SBATCH --nodelist=vrhpc4.desic.upv.es

source ~/herramientas/miniconda3/bin/activate tf

python3 ~/test/test.py
