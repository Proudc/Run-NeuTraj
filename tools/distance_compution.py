import pickle
# import traj_dist.distance as  tdist
import os
from distance import distance
import numpy as  np
import multiprocessing
import sys
print(sys.path)

# def trajectory_distance(traj_feature_map, traj_keys,  distance_type = "hausdorff", batch_size = 50, processors = 30):
#     # traj_keys= traj_feature_map.keys()
#     trajs = []
#     for k in traj_keys:
#         traj = []
#         for record in traj_feature_map[k]:
#             traj.append([record[1],record[2]])
#         trajs.append(np.array(traj))

#     pool = multiprocessing.Pool(processes=processors)
#     # print np.shape(distance)
#     batch_number = 0
#     for i in range(len(trajs)):
#         if (i!=0) & (i%batch_size == 0):
#             print (batch_size*batch_number, i)
#             pool.apply_async(trajectory_distance_batch, (i, trajs[batch_size*batch_number:i], trajs, distance_type,
#                                                          'geolife'))
#             batch_number+=1
#     pool.close()
#     pool.join()


# def trajecotry_distance_list(trajs, distance_type = "hausdorff", batch_size = 50, processors = 30, data_name = 'porto' ):
#     pool = multiprocessing.Pool(processes = processors)
#     batch_number = 0
#     for i in range(len(trajs)):
#         if (i != 0) & (i % batch_size == 0):
#             print(batch_size * batch_number, i)
#             pool.apply_async(trajectory_distance_batch, (i, trajs[batch_size*batch_number:i], trajs, distance_type,
#                                                          data_name))
#             batch_number+=1
#     pool.close()
#     pool.join()

# def trajectory_distance_batch(i, batch_trjs, trjs, metric_type = "hausdorff", data_name = 'porto'):
#     if metric_type == 'lcss' or  metric_type == 'edr':
#         trs_matrix = tdist.cdist(batch_trjs, trjs, metric=metric_type, eps= 0.003)
#     # elif metric_type=='erp':
#     #     trs_matrix = tdist.cdist(batch_trjs, trjs, metric=metric_type, eps=0.003)
#     else:
#         trs_matrix = tdist.cdist(batch_trjs, trjs, metric=metric_type)
#     cPickle.dump(trs_matrix, open('./features/'+data_name+'_'+metric_type+'_distance_' + str(i), 'w'))
#     print 'complete: '+str(i)


# def trajectory_distance_combain(trajs_len, batch_size = 100, metric_type = "hausdorff", data_name = 'porto'):
#     distance_list = []
#     a = 0
#     for i in range(1,trajs_len+1):
#         if (i!=0) & (i%batch_size == 0):
#             distance_list.append(cPickle.load(open('./features/'+data_name+'_'+metric_type+'_distance_' + str(i))))
#             print distance_list[-1].shape
#     a = distance_list[-1].shape[1]
#     distances = np.array(distance_list)
#     print distances.shape
#     all_dis = distances.reshape((trajs_len,a))
#     print all_dis.shape
#     cPickle.dump(all_dis,open('./features/'+data_name+'_'+metric_type+'_distance_all_'+str(trajs_len),'w'))
#     return all_dis



def trajectory_all_distance(tr_id, trs_compare_dict):
    print("Begin compute")
    if not os.path.exists("./features/frechet_distance"):
        os.makedirs("./features/frechet_distance")
    trs_matrix = distance.cdist(trs_compare_dict, {tr_id:trs_compare_dict[tr_id]}, type = "frechet")
    pickle.dump(trs_matrix, open('./features/frechet_distance/frechet_distance_'+str(tr_id), 'wb'))
    print("Complete: "+str(tr_id))


def trajectory_diatance_batch(trs_compare_dict, processors = 30):
    pool = multiprocessing.Pool(processes=processors)
    for tr_id in trs_compare_dict:
        pool.apply_async(trajectory_all_distance, (tr_id, trs_compare_dict))
    pool.close()
    pool.join()

def trajectory_distance_combain(trajs_len, inputPath):
    files = os.listdir(inputPath)
    files_index = []
    for fn in files:
        i = int(fn.split('_')[2])
        files_index.append((fn,i))
    files_index.sort(key=lambda x:x[1])
    all_dis = np.zeros((trajs_len, trajs_len))
    for fn in files_index:
        dists = pickle.load(open(inputPath+fn[0],'rb'))
        for key in dists:
            if key[0] < trajs_len and key[1] < trajs_len:
                all_dis[key[0]][key[1]] = dists[key]
    pickle.dump(all_dis, open('./features/toy_discret_frechet_distance_all_1800','wb'))
    return all_dis