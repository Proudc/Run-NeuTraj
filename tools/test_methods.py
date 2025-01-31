import config
import numpy as np
import torch.autograd as autograd
import torch
# from geo_rnns.spatial_memory_lstm_pytorch import SpatialCoordinateRNNPytorch
import time

# pad的目的是对那些长度不到maxlen的轨迹后边补0
def pad_sequence(traj_grids, maxlen = 100, pad_value = 0.0):
    paddec_seqs = []
    for traj in traj_grids:
        pad_r = np.zeros_like(traj[0]) * pad_value
        while (len(traj) < maxlen):
            traj.append(pad_r)
        paddec_seqs.append(traj)
    return paddec_seqs


def test_comput_embeddings(self, spatial_net, max_len, test_batch = 925):
    
    embeddings_list = []
    start_time = time.time()
    total_time = 0
    j = 0
    while j < len(self.pad_trjs):
        padded_trajs = np.array(pad_sequence(self.pad_trjs[j : j + test_batch], maxlen = max_len))
        
        # length = len(self.padded_trajs[j : j + test_batch])
        length = len(padded_trajs)
        
        time1 = time.time()
        if config.recurrent_unit == 'GRU' or config.recurrent_unit == 'SimpleRNN':
            hidden = autograd.Variable(torch.zeros(length, self.target_size), requires_grad=False).cuda()
        else:
            hidden = (autograd.Variable(torch.zeros(length, self.target_size), requires_grad=False).cuda(),
                      autograd.Variable(torch.zeros(length, self.target_size), requires_grad=False).cuda())
        
        out = spatial_net.rnn([autograd.Variable(torch.Tensor(padded_trajs),
                                                 requires_grad = False).cuda(),
                               self.trajs_length[j : j + test_batch]], hidden)
        # embeddings = out.data.cuda().numpy()
        embeddings = out.data
        embeddings_list.append(embeddings)
        time2 = time.time()
        total_time += time2 - time1
        j += test_batch
        if (j % 1000) == 0:
            print(j, total_time)
    print('embedding time of {} trajectories: {}'.format(self.padded_trajs.shape[0], time.time() - start_time, total_time))
    embeddings_list = torch.cat(embeddings_list, dim=0)
    print("size of embeddings_list: ", embeddings_list.size())
    return embeddings_list.cpu().numpy()

def test_model(self, traj_embeddings, test_range, print_batch = 10, similarity = False, r10in50 = False):
    top_1_count,  l_top_1_count  = 0, 0
    top_5_count,  l_top_5_count  = 0, 0
    top_10_count, l_top_10_count = 0, 0
    top_50_count, l_top_50_count = 0, 0
    top10_in_top50_count = 0
    test_traj_num = 0
    range_num = test_range[-1]
    all_true_distance, all_test_distance = [], []
    error_true, error_test, errorr1050 = 0., 0., 0.
    for i in test_range:

        if similarity:
            # This is for the exp similarity
            test_distance = [(j, float(np.exp(-np.sum(np.square(traj_embeddings[i] - e)))))
                             for j, e in enumerate(traj_embeddings)]
            t_similarity = np.exp(-self.distance[i][:len(traj_embeddings)] * config.mail_pre_degree)
            true_distance = list(enumerate(t_similarity))
            learned_distance = list(enumerate(self.distance[i][:len(self.train_seqs)]))

            s_test_distance = sorted(test_distance, key=lambda a: a[1], reverse=True)
            s_true_distance = sorted(true_distance, key=lambda a: a[1], reverse=True)
            s_learned_distance = sorted(learned_distance, key=lambda a: a[1])
        else:
            # This is for computing the distance

            test_distance = [(j, float(np.sum(np.square(traj_embeddings[i] - e))))
                             for j, e in enumerate(traj_embeddings[test_range[-1]:])]
            true_distance = list(enumerate(self.distance[i][test_range[-1]:]))
            
            # test_distance = [(j, float(np.sum(np.square(traj_embeddings[i] - e))))
            #                  for j, e in enumerate(traj_embeddings)]
            # true_distance = list(enumerate(self.distance[i]))
            
            learned_distance = list(enumerate(self.distance[i][:len(self.train_seqs)]))

            s_test_distance = sorted(test_distance, key=lambda a: a[1])
            s_true_distance = sorted(true_distance, key=lambda a: a[1])
            s_learned_distance = sorted(learned_distance, key=lambda a: a[1])

        top1_recall  = [l[0] for l in s_test_distance[:1] if l[0] in [j[0] for j in s_true_distance[:1]]]
        top5_recall  = [l[0] for l in s_test_distance[:5] if l[0] in [j[0] for j in s_true_distance[:5]]]
        top10_recall = [l[0] for l in s_test_distance[:10] if l[0] in [j[0] for j in s_true_distance[:10]]]
        top50_recall = [l[0] for l in s_test_distance[:50] if l[0] in [j[0] for j in s_true_distance[:50]]]
        top10_in_top50 = [l[0] for l in s_test_distance[:10] if l[0] in [j[0] for j in s_true_distance[:50]]]

        top_1_count += len(top1_recall)
        top_5_count += len(top5_recall)
        top_10_count += len(top10_recall)
        top_50_count += len(top50_recall)
        top10_in_top50_count += len(top10_in_top50)

        l_top10_recall = [l[0] for l in s_learned_distance[:11] if l[0] in [j[0] for j in s_true_distance[:11]]]
        l_top50_recall = [l[0] for l in s_learned_distance[:51] if l[0] in [j[0] for j in s_true_distance[:51]]]

        l_top_10_count += len(l_top10_recall) - 1
        l_top_50_count += len(l_top50_recall) - 1

        all_true_distance.append(s_true_distance[:50])
        all_test_distance.append(s_test_distance[:50])

        true_top_10_distance = 0.
        for ij in s_true_distance[:11]:
            true_top_10_distance += self.distance[i][ij[0]]
        test_top_10_distance = 0.
        for ij in s_test_distance[:11]:
            test_top_10_distance += self.distance[i][ij[0]]
        test_top_10_distance_r10in50 = 0.
        temp_distance_in_test50 = []
        for ij in s_test_distance[:51]:
            temp_distance_in_test50.append([ij,self.distance[i][ij[0]]])
        sort_dis_10in50 = sorted(temp_distance_in_test50, key= lambda x: x[1])
        test_top_10_distance_r10in50 = sum([iaj[1] for iaj in sort_dis_10in50[:11]])


        error_true += true_top_10_distance
        error_test += test_top_10_distance
        errorr1050 += test_top_10_distance_r10in50

        test_traj_num += 1
        # if (i % print_batch) == 0:
        #     # print test_distance
        #     print('**----------------------------------**')
        #     print(s_test_distance[:20])
        #     print(s_true_distance[:20])
        #     print(top10_recall)
        #     print(top50_recall)

    if r10in50:
        error_test = errorr1050

    print('Test on {} trajs'.format(test_traj_num))
    print('Search Top  1 recall {}'.format(float(top_1_count) / (test_traj_num * 1)))
    print('Search Top  5 recall {}'.format(float(top_5_count) / (test_traj_num * 5)))
    print('Search Top 10 recall {}'.format(float(top_10_count) / (test_traj_num * 10)))
    print('Search Top 50 recall {}'.format(float(top_50_count) / (test_traj_num * 50)))
    print('Search Top 10 in Top 50 recall {}'.format(float(top10_in_top50_count) / (test_traj_num * 10)))
    print('Error true:{}'.format((float(error_true) / (test_traj_num * 10))*84000))
    print('Error test:{}'.format((float(error_test) / (test_traj_num * 10))*84000))
    print('Error div :{}'.format((float(abs(error_test-error_true)) / (test_traj_num * 10))*84000))
    return (float(top_10_count) / (test_traj_num * 10), \
           float(top_50_count) / (test_traj_num * 50),\
           float(top10_in_top50_count) / (test_traj_num * 10), \
           (float(error_true) / (test_traj_num * 10)) * 84000, \
           (float(error_test) / (test_traj_num * 10)) * 84000, \
           (float(abs(error_test - error_true)) / (test_traj_num * 10)) * 84000)


if __name__ == '__main__':
    print(config.config_to_str())