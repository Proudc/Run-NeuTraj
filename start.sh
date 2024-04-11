# python preprocessing.py

# python train.py 0_porto_10000 haus 666
# python train.py 0_geolife haus 666


dataset_list=(0_porto_all)
dist_type_list=(frechet haus dtw edr)
seed_list=(666 555 444)
for dataset in ${dataset_list[@]}; do
    for dist_type in ${dist_type_list[@]}; do
        for seed in ${seed_list[@]}; do
            train_flag=${dataset}_${dist_type}_${seed}
            echo ${train_flag}
            nohup python train.py ${dataset} ${dist_type} ${seed} > train_log/${train_flag} &
            PID0=$!
            wait $PID0
        done
    done
done

