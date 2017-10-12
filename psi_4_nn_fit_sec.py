# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 14:13:59 2017

@author: ray
"""

import numpy as np
import sys
import os
import time
import math
from sklearn import linear_model
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Activation
import keras
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.axes3d as p3

import itertools
import multiprocessing

try:
    import cPickle as pickle
except:
    import pickle
import matplotlib.pyplot as plt
from subsampling import subsampling_system, random_subsampling, subsampling_system_with_PCA


def map_to_n1_1(arr, maxx, minn):
    return np.divide(np.subtract(arr, minn), (maxx - minn) / 2.) - 1.


def log(log_filename, text):
    with open(log_filename, "a") as myfile:
        myfile.write(text)
    return


def map_to_0_1(arr, maxx, minn):
    return np.divide(np.subtract(arr, minn), (maxx - minn))


def map_back(arr, maxx, minn):
    return np.add(np.multiply(arr, (maxx - minn)), minn)


def get_start_loss(log_filename):
    with open(log_filename, 'r') as f:
        for line in f:
            pass
        temp = line

    if temp.strip().startswith('updated'):
        return float(temp.split()[2])
    else:
        raise ValueError



def fit_with_KerasNN(X, y, functional, target, gamma, num_desc_deri, num_desc_deri_squa, num_desc_ave_dens,
                     desc_transform, target_transform, lower, upper, n_per_layer, n_layers, activation, tol,
                     slowdown_factor, early_stop_trials, ensemble_no):
    #    colors = map_to_0_1(y, max(y),min(y))
    #    fig = plt.figure()
    #    ax = p3.Axes3D(fig)
    #    ax.scatter(X[:,0],X[:,1],y,  c=colors, cmap='hsv', linewidth = 0, alpha=1,s=3)
    ##    ax.scatter(X_train[:,0],X_train[:,1],y_train,linewidth = 0, alpha=1,s=3)
    ##    ax.set_ylim(-10.,10.)
    #    plt.savefig('error_test2.png')
    #    plt.show()


    filename = 'NN_{}_linear_residual_{}_{}_{}_{}_gamma{}_dev{}_devsq{}_inte{}_{}_{}_{}_{}_nn_{}.h5'.format(functional,
                                                                                                      n_per_layer,
                                                                                                      n_layers,
                                                                                                      activation,
                                                                                                      target, gamma,
                                                                                                      num_desc_deri,
                                                                                                      num_desc_deri_squa,
                                                                                                      num_desc_ave_dens,
                                                                                                      desc_transform,
                                                                                                      target_transform,
                                                                                                      str(int(
                                                                                                          lower)).replace(
                                                                                                          '-',
                                                                                                          'n').replace(
                                                                                                          '.', '-'),
                                                                                                      str(int(
                                                                                                          upper)).replace(
                                                                                                          '-',
                                                                                                          'n').replace(
                                                                                                          '.', '-'), ensemble_no)
    log_filename = 'NN_{}_linear_residual_{}_{}_{}_{}_gamma{}_dev{}_devsq{}_inte{}_{}_{}_{}_{}_nn_{}_log.log'.format(
        functional, n_per_layer, n_layers, activation, target, gamma, num_desc_deri, num_desc_deri_squa,
        num_desc_ave_dens, desc_transform, target_transform, str(int(lower)).replace('-', 'n').replace('.', '-'),
        str(int(upper)).replace('-', 'n').replace('.', '-'),ensemble_no)
    try:
        model = load_model(filename)
        restart = True
        print 'model loaded: ' + filename
    except:
        restart = False
        n = int(n_per_layer)
        k = len(X[0])
        print n, k

        model = Sequential()

        model.add(Dense(output_dim=n, input_dim=k, activation=activation))

        if n_layers > 1:
            for i in range(int(n_layers - 1)):
                model.add(Dense(input_dim=n, output_dim=n, activation=activation))
        model.add(Dense(input_dim=n, output_dim=1, activation='linear'))

    # model.add(Dense(input_dim = 1,output_dim =1, activation = 'linear',  init='uniform'))

    print 'model set'
    default_lr = 0.001
    adam = keras.optimizers.Adam(lr=default_lr / slowdown_factor)
    model.compile(loss='mse',  # custom_loss,
                  optimizer=adam)
    # metrics=['mae'])
    print model.summary()
    print model.get_config()

    est_start = time.time()
    history_callback = model.fit(X, y, nb_epoch=1, batch_size=500000)
    est_epoch_time = time.time() - est_start
    if est_epoch_time >= 30.:
        num_epoch = 1
    else:
        num_epoch = int(math.floor(30. / est_epoch_time))
    if restart == True:
        try:
            start_loss = get_start_loss(log_filename)
        except:
            loss_history = history_callback.history["loss"]
            start_loss = np.array(loss_history)[0]
    else:
        loss_history = history_callback.history["loss"]
        start_loss = np.array(loss_history)[0]

    log(log_filename, "\n start: " + str(start_loss))

    old_loss = start_loss
    keep_going = True

    count_epochs = 0
    while keep_going:
        count_epochs += 1
        history_callback = model.fit(X, y, nb_epoch=num_epoch, batch_size=500000, shuffle=True)
        loss_history = history_callback.history["loss"]
        new_loss = np.array(loss_history)[-1]
        if new_loss < old_loss:
            model.save(filename)
            print 'model saved'
            log(log_filename,
                "\n updated best: " + str(new_loss) + " \t epochs since last update: " + str(count_epochs))
            old_loss = new_loss
            count_epochs = 0
        if new_loss < tol:
            keep_going = False
        if count_epochs >= early_stop_trials:
            keep_going = False

            ##    plt.scatter(X, y-model.predict(X),  color='black')
            #
            #
            #    plt.scatter(X[:,0], y,  color='black')
            #    plt.scatter(X[:,0], model.predict(X), color='blue',
            #             linewidth=3)
            #    test_x = np.linspace(-6., 5., 10000)
            #    plt.plot(test_x, model.predict(test_x),color='blue')
            #    plt.show()
    return model


if __name__ == "__main__":

    list_molecule_filename = sys.argv[1]
    functional = sys.argv[2]
    h = float(sys.argv[3])
    L = float(sys.argv[4])
    N = int(sys.argv[5])
    gamma = int(sys.argv[6])
    num_desc_deri = int(sys.argv[7])
    num_desc_deri_squa = int(sys.argv[8])
    num_desc_ave_dens = int(sys.argv[9])
    target = sys.argv[10]
    desc_transform = sys.argv[11]
    target_transform = sys.argv[12]
    lower = float(sys.argv[13])
    upper = float(sys.argv[14])
    n_per_layer = int(sys.argv[15])
    n_layers = int(sys.argv[16])
    activation_choice = sys.argv[17]
    slowdown_factor = float(sys.argv[18])
    tol = float(sys.argv[19])
    ensemble_size = int(sys.argv[20])
    try:
        early_stop_trials = int(sys.argv[21])
    except:
        early_stop_trials = 100

    if activation_choice not in ['tanh', 'relu', 'sigmoid', 'softmax']:
        raise ValueError

    if desc_transform not in ['log', 'real']:
        raise ValueError

    if target_transform not in ['log', 'real', 'negreal']:
        raise ValueError

    # if dataset_choice not in ['all','dens']:
    #    raise ValueError

    # print device_lib.list_local_devices()
    cwd = os.getcwd()
    result_dir = "{}_{}_{}_{}_{}_gamma{}_dev{}_devsq{}_inte{}_{}_{}_models".format(functional, str(L).replace('.', '-'),
                                                                                   str(h).replace('.', '-'), N, target,
                                                                                   gamma, num_desc_deri,
                                                                                   num_desc_deri_squa,
                                                                                   num_desc_ave_dens, desc_transform,
                                                                                   target_transform)
    if os.path.isdir(result_dir) == False:
        os.makedirs(cwd + '/' + result_dir)

    #y, dens = get_training_data(list_molecule_filename, functional, h, L, N, target, gamma, num_desc_deri,
                                        # num_desc_deri_squa, num_desc_ave_dens, desc_transform, target_transform, lower,
                                        # upper)

    os.chdir(cwd + '/' + result_dir)

    filename = 'Linear_{}_model_{}_gamma{}_dev{}_devsq{}_inte{}_{}_{}_{}_{}.p'.format(functional, target, gamma,
                                                                                      num_desc_deri,
                                                                                      num_desc_deri_squa,
                                                                                      num_desc_ave_dens,
                                                                                      desc_transform,
                                                                                      target_transform,
                                                                                      str(int(lower)).replace('-',
                                                                                                              'n').replace(
                                                                                          '.', '-'),
                                                                                      str(int(upper)).replace('-',
                                                                                                              'n').replace('.', '-'))
                                                                        
    print(filename)
    training_data = np.array(pickle.load(open(filename,'rb')));
    print training_data.shape
    for ensemble_no in np.arange(ensemble_size):
    #80 % data for now
        subset_size = int(training_data.shape[0]*0.8)
        training_data_subset = np.random.permutation(training_data)[:subset_size]

        residual = training_data_subset[:,0]
        X_train = training_data_subset[:,1:]

        result_dir2 = result_dir+'_nn_{}'.format(ensemble_no)

        cwd2 = os.getcwd();

        if os.path.isdir(result_dir2) == False:
            os.makedirs(cwd2 + '/' + result_dir2)

        os.chdir(cwd2 + '/' + result_dir2)

        data_fname = "data_nn_{}.p".format(ensemble_no)

        result_list = []
        result_list.append(residual.tolist())
        result_list.append(X_train.tolist())
        result = zip(*result_list)

        with open(data_fname, 'wb') as handle:
            pickle.dump(result, handle, protocol=2)
        
        #print X_train.reshape((257,2))
        
        print residual.shape
        
        model = fit_with_KerasNN(X_train, residual, functional, target, gamma, num_desc_deri, num_desc_deri_squa,
                             num_desc_ave_dens, desc_transform, target_transform, lower, upper, n_per_layer, n_layers,
                             activation_choice, tol, slowdown_factor, early_stop_trials, ensemble_no)
        os.chdir(cwd2)

    os.chdir(cwd)






