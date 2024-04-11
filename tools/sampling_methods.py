import random
import numpy as np
import config
import pickle

np.random.seed(2018)

def random_sampling(train_seq_len, index):
    sampling_index_list = random.sample(range(train_seq_len), config.sampling_num)
    return sampling_index_list

def distance_sampling(distance, train_seq_len, index):
    index_dis = distance[index]
    pre_sort = [np.exp(-i * config.mail_pre_degree) for i in index_dis[:train_seq_len]]

    

    t = 0
    importance = []
    for i in pre_sort/np.sum(pre_sort):
        importance.append(t)
        t += i
    importance = np.array(importance)
    sample_index = []
    generate_count = 0
    while len(sample_index) < config.sampling_num:
        a = np.random.uniform()
        idx = np.where(importance > a)[0]
        
        # print(len(idx), idx[0] - 1, sample_index, idx)
        if len(idx) == 0:
            sample_index.append(train_seq_len-1)
        elif ((idx[0] - 1) not in sample_index) & (not ((idx[0] - 1) == index)):
            sample_index.append(idx[0] - 1)

        if len(sample_index) == 0:
            generate_count += 1
            if generate_count >= 10:
                sorted_list = []
                for i in range(train_seq_len):
                    sorted_list.append((i, distance[index][i]))
                sorted_list = sorted(sorted_list, key= lambda a:a[1], reverse=False)
                sample_index = [i[0] for i in sorted_list[:config.sampling_num]]
                break
        
    sorted_sample_index = []
    for i in sample_index:
        sorted_sample_index.append((i, pre_sort[i]))
    sorted_sample_index = sorted(sorted_sample_index, key= lambda a:a[1], reverse=True)
    return [i[0] for i in sorted_sample_index]

def negative_distance_sampling(distance, train_seq_len, index):
    index_dis = distance[index]
    pre_sort = [np.exp(-i * config.mail_pre_degree) for i in index_dis[:train_seq_len]]
    pre_sort = np.ones_like(np.array(pre_sort)) - pre_sort
    # print [(i,j) for i,j in enumerate(pre_sort)]
    sample_index = []
    t = 0
    importance = []
    for i in pre_sort / np.sum(pre_sort):
        importance.append(t)
        t += i
    importance = np.array(importance)
    # print importance
    while len(sample_index) < config.sampling_num:
        a = np.random.uniform()
        idx = np.where(importance > a)[0]
        if len(idx) == 0:
            sample_index.append(train_seq_len - 1)
        elif ((idx[0] - 1) not in sample_index) & (not ((idx[0] - 1) == index)):
            sample_index.append(idx[0] - 1)
    sorted_sample_index = []
    for i in sample_index:
        sorted_sample_index.append((i, pre_sort[i]))
    sorted_sample_index = sorted(sorted_sample_index, key=lambda a: a[1], reverse=True)
    return [i[0] for i in sorted_sample_index]


def top_n_sampling(distance, train_seq_len, index):
    index_dis = distance[index]

    pre_sort = [(i,j) for i,j in enumerate(index_dis[:train_seq_len])]
    post_sort = sorted(pre_sort, key=lambda k:k[1])
    sample_index = [e[0] for e in post_sort[:config.sampling_num]]
    return sample_index

if __name__ == '__main__':
    distance = pickle.load(open('../features/discret_frechet_distance_all_600', 'rb'))
    print(distance_sampling(distance, 100, 10))
    print(distance_sampling(distance, 100, 10))

    print(negative_distance_sampling(distance, 100, 10))
    print(negative_distance_sampling(distance, 100, 10))

    print(top_n_sampling(distance,100,10))
    print(top_n_sampling(distance,100,10))