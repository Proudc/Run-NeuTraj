# Data path
distance_type = "haus"
data_type = "shorttraj"
corrdatapath = './features/' + data_type + '_traj_coord'
gridxypath = './features/' + data_type + '_traj_grid'
# distancepath = './data/' + data_type + '_' + distance_type + '_train_distance_matrix_result'
# distancepath = './data/dis_matrix_10000x10000.pkl'
distancepath = './data/0_Test_porto10/' + distance_type + '_train_distance_matrix_result'

# Training Prarmeters
GPU = "1"
learning_rate = 0.01
seeds_radio = 0.5259
# seeds_radio = 0.6


# 原始epoch
# epochs = 100000
epochs = 2
# epochs = 2

save_epoch = epochs
batch_size = 128
sampling_num = 10

random_seed = 666

# distance_type = distancepath.split('/')[2].split('_')[1]
# data_type = distancepath.split('/')[2].split('_')[0]


if distance_type == 'dtw':
    mail_pre_degree = 16
else:
    mail_pre_degree = 8




# Test Config
datalength = 5704
# datalength = 10000

em_batch = 304
test_num = 100

# Model Parameters
d = 128
stard_unit = False # It controls the type of recurrent unit (standrad cells or SAM argumented cells)
incell = True
recurrent_unit = 'GRU' #GRU, LSTM or SimpleRNN
spatial_width  = 2

# gird_size = [1100, 1100]
gird_size = [3500, 7000]




def config_to_str():
   configs = '********************************Config Begin********************************\n'+\
             'learning_rate = {} '.format(learning_rate)+ '\n'+\
             'mail_pre_degree = {} '.format(mail_pre_degree)+ '\n'+\
             'seeds_radio = {} '.format(seeds_radio) + '\n' +\
             'epochs = {} '.format(epochs)+ '\n'+\
             'datapath = {} '.format(corrdatapath) +'\n'+ \
             'datatype = {} '.format(data_type) + '\n' + \
             'corrdatapath = {} '.format(corrdatapath)+ '\n'+ \
             'distancepath = {} '.format(distancepath) + '\n' + \
             'distance_type = {}'.format(distance_type) + '\n' + \
             'recurrent_unit = {}'.format(recurrent_unit) + '\n' + \
             'batch_size = {} '.format(batch_size)+ '\n'+\
             'sampling_num = {} '.format(sampling_num)+ '\n'+\
             'incell = {}'.format(incell)+ '\n'+ \
             'stard_unit = {}'.format(stard_unit) + '\n' + \
             '********************************Config End********************************\n'
   return configs


if __name__ == '__main__':
    print('../model/model_training_600_{}_acc_{}'.format((0),1))
    print(config_to_str())
