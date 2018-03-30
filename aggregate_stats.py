import matplotlib
matplotlib.use('Agg') 
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.pyplot as plt

import numpy as np
import csv
import sys
import os
import time
import math
import json
from glob import glob
from sklearn import linear_model
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Activation
import keras
import scipy

import itertools
import multiprocessing

try: import cPickle as pickle
except: import pickle
import matplotlib.pyplot as plt
from subsampling import subsampling_system,random_subsampling,subsampling_system_with_PCA


def save_resulting_figure(n,LDA_x,X,NN_model,y):

    predict_y = predict_LDA_residual(n,LDA_x,X,NN_model)

    LDA_predict_y = predict_LDA(n,LDA_x)

    error = y - predict_y

    fig=plt.figure(figsize=(40,40))

    plt.scatter(dens, y,            c= 'red',  lw = 0,label='original',alpha=1.0)
    plt.scatter(dens, predict_y,    c= 'blue',  lw = 0,label='predict',alpha=1.0)
    plt.scatter(dens, error,            c= 'yellow',  lw = 0,label='error',alpha=1.0)

    legend = plt.legend(loc="best", shadow=False, scatterpoints=1, fontsize=30, markerscale=3)

    plt.tick_params(labelsize=60)
    
    plt.savefig('result_plot.png')

    fig=plt.figure(figsize=(40,40))

    plt.scatter(dens, error,            c= 'red',  lw = 0,label='error',alpha=1.0)

    legend = plt.legend(loc="best", shadow=False, scatterpoints=1, fontsize=30, markerscale=3)

    plt.tick_params(labelsize=60)
    
    plt.savefig('error_plot.png')

    csv_result = []
    csv_result.append(dens.tolist())
    csv_result.append(y.tolist())
    csv_result.append(LDA_predict_y.tolist())
    csv_result.append(predict_y.tolist())
    csv_result.append(error.tolist())

    result = np.stack(csv_result,axis=1).tolist()
    with open(result_csv_filename, "wb") as f:
        writer = csv.writer(f)
        writer.writerow(['density','y','LDA_predict_y','predict_y','error'])
        writer.writerows(result)

    return

if __name__ == "__main__":

    setup_filename = sys.argv[1]
    dataset_name = sys.argv[2]
    num_submodel = sys.argv[3]

    with open(setup_filename) as f:
        setup = json.load(f)


    h = float(setup['grid_spacing'])
    L = float(setup['box_dimension'])
    N = int(setup['number_segment_per_side'])
    dir_name = "{}_{}_{}".format(str(L).replace('.','-'),str(h).replace('.','-'),N)

    working_dir = os.getcwd() + '/' + dir_name + '/' + dataset_name

    setup["working_dir"] = working_dir

    model_save_dir = working_dir + "/" + "NN_LDA_residual_{}_{}_{}".format(setup["NN_setup"]["number_neuron_per_layer"], setup["NN_setup"]["number_layers"], setup["NN_setup"]["activation"])

    setup["model_save_dir"] = model_save_dir

    print "Getting data"
    print working_dir
    
    if os.path.isdir(model_save_dir) == False:
        print("Invalid directory")
        raise FileNotFoundError()

    os.chdir(model_save_dir)

    mean_agg = 
    
    #residual,li_model = fit_with_Linear(dens,y)
    #for i in np.arange(no_submodels):
    #size_subsample = int(0.632 * residual.shape[0])
    #subsample_mask = np.random.choice(residual.shape[0], size_subsample)

    #sub_residual = residual[subsample_mask]
    #sub_X = X_train[subsample_mask]
    for submodel_no in np.arange(num_submodels)+1:
        sub_dir = model_save_dir + "/" + "Submodel_{}".format(submodel_no)
        if os.path.isdir(sub_dir) == False:
            print("ERROR: Submodel dir doesn't exist")
            raise FileNotFoundError()

        os.chdir(sub_dir)

        pred_fname =  "predictions_submodel_{}.h5".format(submodel_no)
        hf = h5py.File(pred_fname, 'r')

    #save_resulting_figure(dens,result.x,X_train,NN_model,y)
    
    pred_fname = "predictions_submodel_{}.h5".format(submodel_no)
    hf = h5py(pred_fname, 'w')
    hf.create_dataset('density', data=np.asarray(dens))
    hf.create_dataset('y', data=np.asarray(y))
    hf.create_dataset('LDA_predict_y', data=np.asarray(dens))
    hf.create_dataset('NN_predict_y', data=NN_predict_y)
    hf.create_dataset('predict_y', data=predict_y)
    hf.create_dataset('error', data=error)
    hf.close()
    
    print("Data saved in h5")
    