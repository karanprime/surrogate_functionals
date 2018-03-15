import h5py
import predictLDA_NN


def get_testing_data(dataset_name,setup):

    data_dir_name = setup["working_dir"] + "/data/*/" 
    data_paths = glob(data_dir_name)
    print data_paths


    overall_subsampled_data = []
    overall_random_data = []
    num_samples = len(data_paths)
    num_random_per_molecule = int(math.ceil(float(setup["random_pick"])/float(num_samples)))
    for directory in data_paths:
        temp_molecule_subsampled_data, temp_molecule_random_data = read_data_from_one_dir(directory)
        overall_subsampled_data += temp_molecule_subsampled_data
        overall_random_data += random_subsampling(temp_molecule_random_data, num_random_per_molecule)#no subsampling for testing



    list_subsample = setup["subsample_feature_list"]
    temp_list_subsample = setup["subsample_feature_list"]
    if temp_list_subsample == []:
        for m in range(len(overall_subsampled_data[0])):
            temp_list_subsample.append(m)

    #if len(temp_list_subsample) <= 10:
    #    overall_subsampled_data = subsampling_system(overall_subsampled_data, list_desc = list_subsample, cutoff_sig = float(setup["subsample_cutoff_sig"]), rate = float(setup["subsample_rate"]))
    #else:
    #    overall_subsampled_data = subsampling_system_with_PCA(overall_subsampled_data, list_desc = list_subsample, cutoff_sig = float(setup["subsample_cutoff_sig"]), rate = float(setup["subsample_rate"]),start_trial_component = 9)


    overall = overall_random_data + overall_subsampled_data
    #overall = overall_subsampled_data



    X_train = []
    y_train = []
    dens = []

    for entry in overall:
#        if entry[0] >= lower and entry[0] <= upper:
        X_train.append(list(entry[1:]))
        dens.append(entry[1])
        y_train.append(entry[0])
    
    
    X_train = (np.asarray(X_train))
    y_train = np.asarray(y_train).reshape((len(y_train),1))
    dens = np.asarray(dens).reshape((len(dens),1))
    
    return X_train, y_train, dens

if __name__ == "__main__":

    setup_filename = sys.argv[1]
    dataset_name = sys.argv[2]
    slowdown_factor = float(sys.argv[3])
    tol = float(sys.argv[4])
    try:
        early_stop_trials = int(sys.argv[5])
    except:
        early_stop_trials = 100

    submodel_no = sys.argv[6]

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
    
    X_test,y, dens = get_testing_data(dataset_name,setup)
    
    if os.path.isdir(model_save_dir) == False:
        os.makedirs(model_save_dir)

    os.chdir(model_save_dir)

    residual, result = fit_with_LDA(dens,y)
    #setup['LDA_model'] = result
    
    #residual,li_model = fit_with_Linear(dens,y)
    #for i in np.arange(no_submodels):
    #size_subsample = int(0.632 * residual.shape[0])
    #subsample_mask = np.random.choice(residual.shape[0], size_subsample)

    #sub_residual = residual[subsample_mask]
    #sub_X = X_train[subsample_mask]

    sub_dir = model_save_dir + "/" + "Submodel_{}".format(submodel_no)
    if os.path.isdir(sub_dir) == False:
        print("ERROR: Submodel dir doesn't exist")
        raise FileNotFoundError()

    os.chdir(sub_dir)

    NN_model = fit_with_KerasNN(setup, submodel_no, sub_X, sub_residual, tol, slowdown_factor, early_stop_trials)
    
    print "NN_{} trained".format(submodel_no)
    #save_resulting_figure(dens,result.x,X_train,NN_model,y)
    
    predict_y = predict_LDA_residual(dens,result.x,X_train,NN_model)
    LDA_predict_y = predict_LDA(dens ,result.x)
    error = y - predict_y
    