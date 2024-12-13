from tqdm import tqdm
import torch
import torch.nn.functional as F
#from gclstm_copy import GCLSTM
#from torch_geometric_temporal.dataset import ChickenpoxDatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split
from torch_geometric_temporal.signal import static_graph_temporal_signal, dynamic_graph_temporal_signal
from torch_geometric_temporal.signal import dynamic_graph_temporal_signal_batch
from torch_geometric_temporal.nn.recurrent import GCLSTM, GConvGRU
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
import time
from sklearn.preprocessing import StandardScaler
import random
import torch_geometric
import os
from operator import itemgetter
from datetime import datetime
import pandas as pd
from dateutil.relativedelta import relativedelta

def get_max_shape(data_dict_1, data_dict_2):
    list_of_dicts = []
    list_of_dicts.extend(data_dict_1)
    list_of_dicts.extend(data_dict_2)
    shape_dims = {}
    feat_dims = [list_of_dicts[i]['features'][0].shape[0] for i in range(len(list_of_dicts))]
    edge_dims = [list_of_dicts[i]['edge_index'].shape[1] for i in range(len(list_of_dicts))]
    shape_dims['features'] = max(feat_dims)
    shape_dims['edges'] = max(edge_dims)
    return shape_dims 

def reshape_data_dict_region(data_dict_1, shape_dims):
    zeros_f = np.zeros((shape_dims['features'] - data_dict_1['features'][0].shape[0], data_dict_1['features'][0].shape[1]))
    zeros_e = np.zeros((data_dict_1['edge_index'].shape[0], shape_dims['edges'] - data_dict_1['edge_index'].shape[1]))
    zeros_w = np.zeros(shape_dims['edges'] - data_dict_1['edge_index'].shape[1])
    new_e = np.concatenate((data_dict_1['edge_index'], zeros_e), axis=1)
    new_w = np.concatenate((data_dict_1['edge_weight'], zeros_w), axis=0)
    new_f = []
    for i in range(len(data_dict_1['features'])):
        ff =  np.concatenate((data_dict_1['features'][i], zeros_f), axis=0)
        new_f.append(ff)
    data_dict_transformed = {}
    data_dict_transformed['features'] = new_f
    data_dict_transformed['edge_index'] = new_e
    data_dict_transformed['edge_weight'] = new_w
    data_dict_transformed['targets'] = data_dict_1['targets']
    data_dict_transformed['vics'] = data_dict_1['vics']
    data_dict_transformed['et'] = data_dict_1['et']
    data_dict_transformed['dates'] = data_dict_1['date_consistent_monthly']
    return data_dict_transformed

def sparse_to_dense(edge_index, adj_dim):
    adj = torch.zeros(adj_dim, adj_dim)
    adj[edge_index[0, :], edge_index[1, :]] = 1
    return adj

def load_dataset(name, vars):
    #"models_data_all/"+name+'_processed/'+name+"data_processed_monthly_original.pickle"
    with open('datasets/unified_datasets/'+name+vars+".pickle", 'rb') as h:
        data_dict = pickle.load(h)
    data_dict["et"] = [np.array(et) for et in data_dict["et"]]
    return data_dict

def normalize_outputs(data_dict, train_mean, train_std):
    data_dict["targets"] = (data_dict["targets"] - train_mean)/train_std
    data_dict["targets"] = list(data_dict["targets"])
    data_dict["targets"] = [np.array(data_dict['targets'][i]) for i in range(len(data_dict['targets']))]
    return data_dict

def denormalize_outputs(outputs_list_normalized, train_mean, train_std):
    #outputs_list_denormalized = outputs_list_normalized * train_std + train_mean # n = (d - m) / s # d = ns+m
    outputs_list_denormalized = list(np.array(outputs_list_normalized) * train_std + train_mean)
    return outputs_list_denormalized

def combine_datasets(data_dict_1, data_dict_2, params):
    argmins, argmaxs, mins, maxs, dates, dates_formatted = [], [], [], [], [], []
    data_dict_all = data_dict_1
    data_dict_all.append(data_dict_2)
    for i in range(len(data_dict_all)):
        dates.append(data_dict_all[i]['dates'])
        dates_formatted.append([datetime.strptime(dates[i][j], '%m/%Y') for j in range(len(dates[i]))])
        mins.append(min(dates_formatted[i]))
        maxs.append(max(dates_formatted[i]))
        argmins.append(np.argmin(dates_formatted[i]))
        argmaxs.append(np.argmax(dates_formatted[i]))
    arg_limit_inf = np.argmax(mins)
    arg_limit_sup = np.argmin(maxs)
    limit_inf = mins[arg_limit_inf]
    limit_sup = maxs[arg_limit_sup]
    #
    delta_start = 10 - limit_inf.month
    n_months = 12 * (params['n_years_train']+params['n_years_test']) - 1
    limit_inf_new = limit_inf + relativedelta(months=delta_start)
    limit_sup_new = limit_inf_new + relativedelta(months=n_months)
    #
    #new_date = limit_inf + relativedelta(months=5)
    indexes_all = []
    for i in range(len(dates_formatted)):
        first_idx = dates_formatted[i].index(limit_inf_new)
        last_idx = dates_formatted[i].index(limit_sup_new)
        indexes_all.append([first_idx + j for j in range(last_idx-first_idx+1)])
    #len_adopted = int(len(indexes_all[0]) * 1/7) # we got the indexes we need for all datasets (1 indexset per watershed)
    #indexes_all = [indexes_all[i][:int(len_adopted)] for i in range(len(indexes_all))]
    features_all = []
    et_all = []
    targets_all = []
    dates_all = []
    vic_all = []
    for i in range(len(dates_formatted)):
        features_all.append(list(itemgetter(*indexes_all[i])(data_dict_all[i]["features"])))
        targets_all.append(list(itemgetter(*indexes_all[i])(data_dict_all[i]["targets"])))
        et_all.append(list(itemgetter(*indexes_all[i])(data_dict_all[i]["et"])))
        dates_all.append(list(itemgetter(*indexes_all[i])(data_dict_all[i]["dates"])))
        vic_all.append(list(itemgetter(*indexes_all[i])(data_dict_all[i]["vics"])))
    len_adopted = len(features_all[0])
    train_len = params['n_years_train'] * 12
    n_train_regions = len(features_all) - 1
    features_train = []
    targets_train = []
    et_train = []
    edge_index_train = []
    edge_weight_train = []
    vic_train = []
    for i in range(n_train_regions):
        features_train.extend(np.array_split(features_all[i][:train_len], n_train_regions)[i])
        targets_train.extend(np.array_split(targets_all[i][:train_len], n_train_regions)[i])
        vic_train.extend(np.array_split(vic_all[i][:train_len], n_train_regions)[i])
        et_train.extend(np.array_split(et_all[i][:train_len], n_train_regions)[i])
        chunk_len = len(list(np.array_split(features_all[i][:train_len], n_train_regions)[i]))
        edge_index_train.extend([data_dict_1[i]["edge_index"]] * chunk_len)
        edge_weight_train.extend([data_dict_1[i]["edge_weight"]] * chunk_len)
    data_dict_train = {}
    data_dict_test = {}
    data_dict_train["features"] = features_train
    data_dict_train["targets"] = [np.array(tar) for tar in targets_train]
    data_dict_train["et"] = [np.array(e) for e in et_train]
    data_dict_train["vics"] = [np.array(e) for e in vic_train]
    data_dict_train["date"] = dates_all[0][:train_len]
    data_dict_train["edge_index"] = edge_index_train
    data_dict_train["edge_weight"] = edge_weight_train
    data_dict_test["features"] = features_all[-1][train_len:]
    data_dict_test["targets"] = targets_all[-1][train_len:]
    data_dict_test["et"] = et_all[-1][train_len:]
    data_dict_test["vics"] = [np.array(v) for v in vic_all[-1][train_len:]]
    data_dict_test["date"] = dates_all[0][train_len:]
    data_dict_test["edge_index"] = [data_dict_2["edge_index"]] * (len_adopted-train_len)
    data_dict_test["edge_weight"] = [data_dict_2["edge_weight"]] * (len_adopted-train_len)
    print(data_dict_test["date"][0] + " ... " + data_dict_test["date"][-1])
    return data_dict_train, data_dict_test

def format_dataset(data_dict):
    '''dataset_created = static_graph_temporal_signal.StaticGraphTemporalSignal(edge_index=data_dict["edge_index"], 
                                                                            edge_weight=data_dict["edge_weight"], 
                                                                            features=data_dict["features"], 
                                                                            targets=data_dict["targets"])'''
                                                                            #,sth=[np.array(np.float(i)) for i in range(len(data_dict["targets"]))])
    n_months = len(data_dict["targets"])
    batch_indexes = [np.array(int(i/12)) for i in range(n_months)]
    date_indexes = [np.array(i) for i in range(n_months)]
    dates_np = [np.array(x) for x in data_dict["date"]]
    dataset_created = dynamic_graph_temporal_signal.DynamicGraphTemporalSignal(edge_indices=data_dict["edge_index"], 
                                                                            edge_weights=data_dict["edge_weight"], 
                                                                            features=data_dict["features"], 
                                                                            targets=data_dict["targets"],
                                                                            et=data_dict["et"],
                                                                            vics=data_dict["vics"],
                                                                            date=dates_np,
                                                                            date_index=date_indexes,
                                                                            batches=batch_indexes)
    print("Dataset formatted successfully!")
    return dataset_created

def yearly_batches(data_single):
    n_years = int(len(data_single)/12)
    data_batched = [data_single[i*12:(i+1)*12] for i in range(n_years)]
    return data_batched

class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features, nodes_number, hidden_dimension):
        super(RecurrentGCN, self).__init__()
        #self.recurrent = GCLSTM(node_features, hidden_dimension, 1)
        self.recurrent = GConvGRU(node_features, hidden_dimension, 1)
        self.linear1 = torch.nn.Linear(hidden_dimension, 1)
        self.linear2 = torch.nn.Linear(nodes_number, 1)
        self.LeakyReLU = torch.nn.LeakyReLU()
    #def forward(self, x, edge_index, edge_weight, h, c):
    def forward(self, x, edge_index, edge_weight, h):
        #h_0, c_0 = self.recurrent(x, edge_index, edge_weight, h, c)
        h_0 = h
        h_list= []
        for i in range(len(x)):
            h_0 = self.recurrent(x[i], edge_index, edge_weight, h_0)
            h_1 = torch.unsqueeze(h_0, 0)
            h_list.append(h_1)
        h_tensor = torch.cat(h_list)
        h = self.LeakyReLU(h_tensor)
        h = self.linear1(h)
        h = self.LeakyReLU(h)
        h = torch.squeeze(h) # torch.transpose(h, 0, 1)
        h = torch.squeeze(self.linear2(h))
        #h = 100 * self.LeakyReLU(h)
        #h = self.LeakyReLU(h)
        return h, h_0
        #return h, h_0, c_0

def nse(preds_list, targets_list, mean_all):
    preds = np.array(preds_list)
    targets = np.array(targets_list)
    return (1-(np.sum((preds-targets)**2)/
               np.sum((targets-mean_all)**2))
            )
    
def nnse(preds_list, targets_list, mean_all):
    nse_value = nse(preds_list, targets_list, mean_all)
    return 1/(2 - nse_value)
    
    
def train_RGCN(params):
    train_regions = params['region_train']
    test_region = params['region_test']
    #file_name = 'results_22_dec/results_mixed_tgds_nsl_train_'+ str(len(train_regions))+'_test_'+ test_region+'.txt'
    if params["write_MSE"] or params["write_NSE"] or params["save_wbe"]:
        path = "results_tgds/" + datetime.now().strftime("%Y_%m_%d")
        # Check whether the specified path exists or not
        isExist = os.path.exists(path)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(path)
            print("The new directory is created!")
        if params["write_MSE"]:
            file_name_mse = "results_tgds/" + datetime.now().strftime("%Y_%m_%d") +'/'+datetime.now().strftime("D%Y_%m_%d_T%H_%M") +'_' +params['vars']+'_' +params['region_test']+ '_mse_tgds.txt'
        if params["write_NSE"]:
            file_name_nse = "results_tgds/" + datetime.now().strftime("%Y_%m_%d") +'/'+datetime.now().strftime("D%Y_%m_%d_T%H_%M") +'_' +params['vars']+'_' +params['region_test']+ '_nse_tgds.txt'
        if params["save_wbe"]:
            file_name_wbe = "results_tgds/" + datetime.now().strftime("%Y_%m_%d") +'/'+datetime.now().strftime("D%Y_%m_%d_T%H_%M") +'_' +params['vars']+'_' +params['region_test']+ '_wbe_tgds.csv'
    vars =  params['vars']
    epochs = params['epochs']
    #train_ratio = params['train_ratio']
    train_mix_ratio = params['train_mix_ratio']
    
    train_loaded = [] 
    for i in range(len(train_regions)):
        train_loaded.append(load_dataset(train_regions[i], vars))
    test_loaded = load_dataset(test_region, vars)
    shape_dims = get_max_shape(train_loaded, [test_loaded])
    train_reshaped = []
    for i in range(len(train_loaded)):
        train_reshaped.append(reshape_data_dict_region(train_loaded[i], shape_dims))
    test_reshaped = reshape_data_dict_region(test_loaded, shape_dims)
    data_dict_train, data_dict_test = combine_datasets(train_reshaped, test_reshaped, params)
    train_mean = np.mean(data_dict_train["targets"])
    train_std = np.std(data_dict_train["targets"])
    targets_all = data_dict_train["targets"] + data_dict_test["targets"]
    mean_all = np.mean(targets_all)
    test_date = data_dict_test['date']
    train_normalized = normalize_outputs(data_dict_train, train_mean, train_std)
    test_normalized = normalize_outputs(data_dict_test, train_mean, train_std)
    dataset_train = format_dataset(train_normalized)
    dataset_test = format_dataset(test_normalized)
    model = RecurrentGCN(node_features = dataset_train[0].num_node_features, 
                        nodes_number = dataset_train[0].num_nodes, 
                        hidden_dimension = params["hidden_dim"]) #256
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    train_losses_mse = []
    test_losses_mse = []
    train_preds = []
    test_preds = []
    for epoch in range(epochs):
        st = time.time()
        model.train()
        #loss_real_train = 0
        mse_model_train = 0
        mse_vic_train = 0
        h, c = None, None
        train_targets_epoch = []
        train_preds_epoch = []
        train_vic_epoch = []
        test_targets_epoch = []
        test_preds_epoch = []
        test_vic_epoch = []
        general_train_loss = 0
        dataset_train_list_single = list(enumerate(dataset_train)) # visualize data in list style
        dataset_train_list_batch = yearly_batches(dataset_train_list_single)
        for batch in dataset_train_list_batch:
            batch_feats = [batch[i][1].x for i in range(len(batch))]
            batch_y = torch.cat([torch.unsqueeze(batch[i][1].y,0) for i in range(len(batch))])
            batch_vic = torch.cat([torch.unsqueeze(batch[i][1].vics,0) for i in range(len(batch))])
            batch_output, h = model(batch_feats, batch[0][1].edge_index, batch[0][1].edge_attr, h)
            mse_model_train_batch = F.mse_loss(batch_output*train_std+train_mean, batch_y*train_std+train_mean)
            # loss_tgds
            batch_et = torch.tensor([batch[i][1].et.item() for i in range(len(batch))])
            batch_p = torch.tensor([torch.mean(batch[i][1].x[:,0]).item() for i in range(len(batch))])
            loss_tgds = batch_output + (batch_et-train_mean)/train_std - (batch_p-train_mean)/train_std
            #loss_tgds = batch_output + batch_et - batch_p 
            train_targets_epoch.extend((batch_y).tolist())
            train_preds_epoch.extend((batch_output).tolist())
            train_vic_epoch.extend(((batch_vic-train_mean)/train_std).tolist())
            general_train_loss += mse_model_train_batch + params['lambda_tgds'] * torch.sum(loss_tgds) 
        #print(loss_nsl)
        general_train_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_targets_real = denormalize_outputs(train_targets_epoch, train_mean, train_std)
        train_preds_real = denormalize_outputs(train_preds_epoch, train_mean, train_std)
        train_vics_real = denormalize_outputs(train_vic_epoch, train_mean, train_std)    
        # test
        model.eval()   
        dataset_test_list_single = list(enumerate(dataset_test)) # visualize data in list style
        dataset_test_list_batch = yearly_batches(dataset_test_list_single)
        for batch in dataset_test_list_batch:
            batch_feats = [batch[i][1].x for i in range(len(batch))]
            batch_y = torch.cat([torch.unsqueeze(batch[i][1].y,0) for i in range(len(batch))])
            batch_vic = torch.cat([torch.unsqueeze(batch[i][1].vics,0) for i in range(len(batch))])
            batch_output, h = model(batch_feats, batch[0][1].edge_index, batch[0][1].edge_attr, h)
            test_targets_epoch.extend(batch_y.tolist())
            test_preds_epoch.extend(batch_output.tolist())
            test_vic_epoch.extend(((batch_vic-train_mean)/train_std).tolist())
        # add the difference in sf units and mse 
        # d = ns+m , 
        #if epoch == epochs-1 and params["save_wbe"]: 
        test_targets_real = denormalize_outputs(test_targets_epoch, train_mean, train_std)
        test_preds_real = denormalize_outputs(test_preds_epoch, train_mean, train_std)
        test_vics_real = denormalize_outputs(test_vic_epoch, train_mean, train_std)
        
        train_preds.append(train_preds_epoch)
        test_preds.append(test_preds_epoch) 
        
        mse_train_epoch = F.mse_loss(torch.Tensor(train_targets_real), torch.Tensor(train_preds_real))
        mse_test_epoch = F.mse_loss(torch.Tensor(test_targets_real), torch.Tensor(test_preds_real))
        mse_train_vic = F.mse_loss(torch.Tensor(train_vics_real), torch.Tensor(train_targets_real))
        mse_test_vic = F.mse_loss(torch.Tensor(test_vics_real), torch.Tensor(test_targets_real))
        
        nnse_train_epoch = nnse(train_targets_real, train_preds_real, mean_all)
        nnse_test_epoch = nnse(test_targets_real, test_preds_real, mean_all)
        nnse_train_vic = nnse(train_vics_real, train_targets_real, mean_all)
        nnse_test_vic = nnse(test_vics_real, test_targets_real, mean_all)
        
        et = time.time()   
        if params["print_NSE"]:
            print('Epoch '+str(epoch)+': time: '+str(round(et-st,4)) +"s, train: "+ str(train_regions) +", test: "+ test_region
                + ": train NSEs: {:.4f}".format(nnse_train_epoch.item()) + " vs vic: {:.4f}".format(nnse_train_vic.item())
                + " & test NSEs: {:.4f}".format(nnse_test_epoch.item()) + " vs vic: " + str(round(nnse_test_vic.item(),4)))
            
        if params["print_MSE"]:
            print('Epoch '+str(epoch)+': time: '+str(round(et-st,4)) +"s, train: "+ str(train_regions) +", test: "+ test_region
                + ": train MSEs: {:.4f}".format(mse_train_epoch.item()) + " vs vic: {:.4f}".format(mse_train_vic.item())
                + " & test MSEs: {:.4f}".format(mse_test_epoch.item()) + " vs vic: " + str(round(mse_test_vic.item(),4)))
        
        if params["print_NSE"]==False and params["print_MSE"] == False:
            print('Epoch '+str(epoch)+': time: '+str(round(et-st,4)) +"s, train: "+ str(train_regions) +", test: "+ test_region)
            
        if params["write_NSE"]:
            f = open(file_name_nse, "a")  # append mode
            f.write('Epoch '+str(epoch)+': time: '+str(round(et-st,4)) +"s, train: "+ str(train_regions) +", test: "+ test_region
              + ": train NSEs: {:.4f}".format(nnse_train_epoch.item()) + " vs vic: {:.4f}".format(nnse_train_vic.item())
              + " & test NSEs: {:.4f}".format(nnse_test_epoch.item()) + " vs vic: " + str(round(nnse_test_vic.item(),4))
                + "\n")
            f.close()
        if params["write_MSE"]:
            f = open(file_name_mse, "a")  # append mode
            f.write('Epoch '+str(epoch)+': time: '+str(round(et-st,4)) +"s, train: "+ str(train_regions) +", test: "+ test_region
              + ": train MSEs: {:.4f}".format(mse_train_epoch.item()) + " vs vic: {:.4f}".format(mse_train_vic.item())
              + " & test MSEs: {:.4f}".format(mse_test_epoch.item()) + " vs vic: " + str(round(mse_test_vic.item(),4))
                + "\n")
            f.close()
            
    if params["save_wbe"]:
        wbe_vars = {'targets': test_targets_real,
                    'preds_IW': test_preds_real,
                    'vic': test_vics_real,
                    'dates': test_date}
        df = pd.DataFrame(wbe_vars)
        df.to_csv(file_name_wbe, index=False)
        
    model_results = {}
    model_results['train_targets'] = train_targets_epoch
    model_results['train_preds'] = train_preds
    #model_results['train_vics'] = train_vics
    model_results['test_targets']  = test_targets_epoch
    model_results['test_preds']    = test_preds
    #model_results['test_vics'] = test_vics
    model_results['train_mse'] = train_losses_mse
    model_results['test_mse'] = test_losses_mse
    return model_results

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

results_directory = "results_tgds"
# Check whether the specified path exists or not
isExist = os.path.exists(results_directory)
if not isExist:
   os.makedirs(results_directory)
   print("The new directory is created!")

'''regions = ['Boise_river',  'Clearwater', 'Clearwater_Canyon_Ranger', 
              'Flathead', 'North_Santiam', 'Pine_creek', 'Salmon_river', 
              'South_Fork_clearwater', 'St_Joe_river', 'Yakima']'''
              
regions_dicts_all = [{'region':'Clearwater', 'years_train':26, 'years_test':9, 'epochs':100 ,'lambda_loss':0.1, 'hidden':256, 'lr':0.001, 'wd':0.005}]

regions_dicts = regions_dicts_all

for r in range(len(regions_dicts)):
    params = {"region_train": [regions_dicts[r]['region']],
    "region_test": regions_dicts[r]['region'],
    "vars": "ET_spatiotemp_STL",
    "epochs": regions_dicts[r]['epochs'], #regions_dicts[r]['epochs'],
    'n_years_train':regions_dicts[r]['years_train'], #6
    'n_years_test': regions_dicts[r]['years_test'], #1
    'hidden_dim':regions_dicts[r]['hidden'], #256
    "train_mix_ratio":None,
    "lambda_tgds":regions_dicts[r]['lambda_loss'], #0.1
    "lr":regions_dicts[r]['lr'], #0.001
    "weight_decay":regions_dicts[r]['wd'], #0.005
    "print_NSE":True,
    "print_MSE":False,
    "write_NSE":True,
    "write_MSE":False,
    "save_wbe":True
    }
    
    model_results = train_RGCN(params)
    
    '''
    with open('results_mixed/RGCN_mixed_'+params["vars"]+'_train_'+params['region_train'] +
    '_test_'+params['region_test']+'_mix_'+str(params["train_mix_ratio"]).replace(".", "p")
    +'.pickle', 'wb') as h:
    pickle.dump(results_RGCN_1, h)'''
    

# smaller data
# update lr
# %0s in weigths
# standardize inputs

print("finished!")


