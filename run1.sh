#/bin/bash
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
dataset=$1
a=1
input=~/data/${dataset}data.mat
echo $input
export CUDA_VISIBLE_DEVICES="0"; (python run.py $input result/uniform_${dataset}_a$a.mat -s 0 -a $a;  python run_joint.py $input result/cluster_popular_joint_${dataset}_m1_a$a.mat -s 3 -m 1 -a $a ) &

export CUDA_VISIBLE_DEVICES="1"; (python run.py $input result/popular_${dataset}_m1_a$a.mat -s 1 -m 1 -a $a; python run_joint.py $input result/cluster_uniform_joint_${dataset}_a$a.mat -s 2 -a $a ) &
export CUDA_VISIBLE_DEVICES="2"; python run.py $input result/cluster_uniform_${dataset}_a$a.mat -s 2 -a $a &
export CUDA_VISIBLE_DEVICES="3"; python run.py $input result/cluster_popular_${dataset}_m1_a$a.mat -s 3 -m 1 -a $a &

wait
