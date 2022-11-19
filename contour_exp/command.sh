#!/bin/bash

# Add modules in project root path
this_script_path=$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P) # Change location to current script
root_path=${this_script_path}/../framework
export PYTHONPATH=$PYTHONPATH:$root_path
echo "Project root added to Python path: '${root_path}'"

data="min"
#cifar
MY_PYTHON=python3
MEM_ITER=10
EPOCH=1 #10
TEST_FREQ=20 #50
MEM_SIZE=100
#timestamp=$( date +%T)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
for AUG in  "None" "Both" #"Mem" "Incoming"
do
exp="RUN_E"$EPOCH"_Iter"$MEM_ITER"_M"$MEM_SIZE"_"$AUG$TIMESTAMP
w1_model_name="w1t_"$exp
w2_model_name="w2t_"$exp
w2FT_model_name="w2t_FT_"$exp

$MY_PYTHON ../framework/util/train_model.py --data $data --epochs $EPOCH --tasks 1 --buffer --buf_size $MEM_SIZE --save $w1_model_name  --memIter $MEM_ITER --det
$MY_PYTHON ../framework/util/train_model.py --data $data --epochs $EPOCH --tasks 2 --buffer --init $w1_model_name --save $w2_model_name --save_path --test $TEST_FREQ  --memIter $MEM_ITER --scraug $AUG --det
$MY_PYTHON ../framework/util/train_model.py --data $data --epochs $EPOCH --tasks 2 --init $w1_model_name --save $w2FT_model_name  --det   #--memIter $MEM_ITER --buf_name "new_buff"

grid_size=50 #0 #0 #50
samples=100 #0 #0 #100

$MY_PYTHON mode_connectivity.py $w1_model_name $w2FT_model_name $w2_model_name --data $data --tasks 12 --buffer $w1_model_name --path $w2_model_name --save test --grid $grid_size --size $samples

$MY_PYTHON mode_connectivity_plot.py test_${grid_size}_${samples} --data $data --path $w2_model_name --exp $exp --aug_type $AUG
done