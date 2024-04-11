
import sys
sys.path.append("/home/czh/clustering/trajectory/code/python/NeuTraj-master/tools/")
sys.path.append("/home/czh/clustering/trajectory/code/python/NeuTraj-master/geo_rnns/")

from geo_rnns.neutraj_trainer import NeuTrajTrainer
from tools import config
import os
import random
import numpy as np
import torch

if __name__ == '__main__':
    print('os.environ["CUDA_VISIBLE_DEVICES"]= {}'.format(os.environ["CUDA_VISIBLE_DEVICES"]))
    print(config.config_to_str())

    config.data_type     = sys.argv[1]
    config.distance_type = sys.argv[2]
    config.random_seed   = int(sys.argv[3])


    config.corrdatapath = './features/' + config.data_type + '_traj_coord'
    config.gridxypath = './features/' + config.data_type + '_traj_grid'
    config.distancepath = '/mnt/data_hdd1/czh/Neutraj/' + config.data_type + '/' + config.distance_type + '_train_distance_matrix_result'

    print("---------------------------------------")
    print("    config.data_type: ", config.data_type)
    print("config.distance_type: ", config.distance_type)
    print("  config.random_seed: ", config.random_seed)
    print(" config.corrdatapath: ", config.corrdatapath)
    print("   config.gridxypath: ", config.gridxypath)
    print(" config.distancepath: ", config.distancepath)
    print("---------------------------------------")
    

    if config.data_type == "0_geolife":
        config.seeds_radio = 0.2241147467503361
        config.datalength = 13386
        config.gird_size = [1100, 1100]
    elif config.data_type == "0_porto_10000":
        config.seeds_radio = 0.3
        config.datalength = 10000
        config.gird_size = [3500, 7000]
    elif config.data_type == "0_porto_all":
        config.seeds_radio = 0.001873151433
        config.datalength = 1601579
        config.gird_size = [1100, 1100]


    else:
        exit()

    
    if config.distance_type == 'dtw':
        config.mail_pre_degree = 16
    else:
        config.mail_pre_degree = 8
    

    random.seed(config.random_seed)
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.random_seed)
    trajrnn = NeuTrajTrainer(tagset_size  = config.d,
                             batch_size   = config.batch_size,
                             sampling_num = config.sampling_num)
    trajrnn.data_prepare(griddatapath = config.gridxypath,
                         coordatapath = config.corrdatapath,
                         distancepath = config.distancepath,
                         train_radio  = config.seeds_radio)
    trajrnn.neutraj_train(load_model = None,
                          in_cell_update = config.incell,
                          stard_LSTM = config.stard_unit,
                          test_model = False)
