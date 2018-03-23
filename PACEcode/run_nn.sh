#!/bin/bash
#PYFILE="trainLDAnn.py"
PYFILE="predictLDAnn.py"
#PYFILE="analyzeLDAnn.py"
NNSETUP="10_0-02_5_LDA_residual_fit_20_2_tanh_setup.json"
for DATASET in "epxc_mGGA_real_real_numerical"; do
    for SUBMODEL in {1..10}; do
#     for NNSETUP in "10_0-02_5_LDA_residual_fit_20_1_relu_setup.json" "10_0-02_5_LDA_residual_fit_20_1_tanh_setup.json" \
# "10_0-02_5_LDA_residual_fit_20_2_relu_setup.json" "10_0-02_5_LDA_residual_fit_20_2_tanh_setup.json" \
# "10_0-02_5_LDA_residual_fit_40_1_relu_setup.json" "10_0-02_5_LDA_residual_fit_40_1_tanh_setup.json" \
# "10_0-02_5_LDA_residual_fit_40_2_relu_setup.json" "10_0-02_5_LDA_residual_fit_40_2_tanh_setup.json" \
# "10_0-02_5_LDA_residual_fit_60_1_relu_setup.json" "10_0-02_5_LDA_residual_fit_60_1_tanh_setup.json" \
# "10_0-02_5_LDA_residual_fit_60_2_relu_setup.json" "10_0-02_5_LDA_residual_fit_60_2_tanh_setup.json" \
# "10_0-02_5_LDA_residual_fit_80_1_relu_setup.json" "10_0-02_5_LDA_residual_fit_80_1_tanh_setup.json" \
# "10_0-02_5_LDA_residual_fit_80_2_relu_setup.json" "10_0-02_5_LDA_residual_fit_80_2_tanh_setup.json" \
# "10_0-02_5_LDA_residual_fit_100_1_relu_setup.json" "10_0-02_5_LDA_residual_fit_100_1_tanh_setup.json" \
# "10_0-02_5_LDA_residual_fit_100_2_relu_setup.json" "10_0-02_5_LDA_residual_fit_100_2_tanh_setup.json"; do
        qsub -v PYFILE=$PYFILE,DATASET=$DATASET,NNSETUP=$NNSETUP,SLOW=100,STOP=500,NSUBMODEL=$SUBMODEL array_run_fit_cpu_LDA_residual.sh
    done
done
