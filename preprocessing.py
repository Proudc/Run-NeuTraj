from tools.distance_compution import trajectory_diatance_batch, trajectory_distance_combain
from tools import preprocess
import pickle
import numpy as  np


def distance_comp(coor_path):
    traj_coord = pickle.load(open(coor_path, 'rb'))[0]
    # print(traj_coord)
    np_traj_coord = [np.array(t) for t in traj_coord]
    
    print(np_traj_coord[0])
    print(np_traj_coord[1])
    print(len(np_traj_coord))

    traj_dict = {}
    for i, traj in enumerate(np_traj_coord):
        traj_dict[i] = traj
    
    # trajectory_diatance_batch(traj_dict, processors = 96)
    # trajectory_distance_combain(1800, "/home/changzhihao/clustering/trajectory/code/python/NeuTraj/features/frechet_distance/")

if __name__ == '__main__':

    # 0_porto_10000
    porto_10000_lon_range = [-13.172526, -6.724476]
    porto_10000_lat_range = [39.256344, 42.124779]
    
    # 0_porto_all
    # porto_all_lon_range = [-13.172526, -6.724476]
    # porto_all_lat_range = [39.256344, 42.124779]
    porto_lat_range = [41.10, 41.24]
    porto_lon_range = [-8.73, -8.5]

    
    # 0_geolife
    geolife_lon_range = [116.200047, 116.499288]
    geolife_lat_range = [39.851057, 40.0699999]
    
    # coor_path, data_name = preprocess.trajectory_feature_generation(path= './data/testporto_trajs')
    coor_path, data_name = preprocess.trajectory_feature_generation(path= './data/0_porto_all/traj_list', lat_range = porto_lat_range, lon_range = porto_lon_range)
    # distance_comp('./features/toy_traj_coord')
