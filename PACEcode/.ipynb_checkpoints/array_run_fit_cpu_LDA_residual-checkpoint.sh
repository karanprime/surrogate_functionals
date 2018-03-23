#!/bin/bash
#PBS -l nodes=1:ppn=1
#PBS -l walltime=48:00:00
#PBS -l pmem=36gb
#PBS -q joe
#PBS -N fit_${DATASET}
#PBS -o stdout_fit_${DATASET}
#PBS -e stderr_fit_${DATASET}
#PBS -M kshah84@mail.gatech.edu

cd $kxc
module purge
module load use.own
module load anaconda/2-4.2.0
source activate psi4-fd-cpu

python ./code/${PYFILE} ${NNSETUP} ${DATASET} ${SLOW}. 1e-10 ${STOP} ${NSUBMODEL}
