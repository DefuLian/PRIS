#/bin/bash
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
dataset=gowalla
input=~/data/${dataset}data.mat
echo $input

export CUDA_VISIBLE_DEVICES="0"
neg=15
(
for a in 5 2 1 0.5 0.25
  do
    python run.py $input result1/uniform_${dataset}_nw_n${neg}_a${a}.mat -s 0 -n $neg -w -a $a
    python run.py $input result1/uniform_${dataset}_n${neg}_a${a}.mat -s 0 -n $neg -a $a
    python run_joint.py $input result1/joint_${dataset}_n${neg}_a${a}.mat -s 2 -m 1 -n $neg -a $a
  done
) &
  
export CUDA_VISIBLE_DEVICES="1"
neg=20
(
for a in 5 2 1 0.5 0.25
  do
    python run.py $input result1/uniform_${dataset}_nw_n${neg}_a${a}.mat -s 0 -n $neg -w -a $a
    python run.py $input result1/uniform_${dataset}_n${neg}_a${a}.mat -s 0 -n $neg -a $a
    python run_joint.py $input result1/joint_${dataset}_n${neg}_a${a}.mat -s 2 -m 1 -n $neg -a $a
  done
) &
  
export CUDA_VISIBLE_DEVICES="2"
neg=25
(
for a in 5 2 1 0.5 0.25
  do
    python run.py $input result1/uniform_${dataset}_nw_n${neg}_a${a}.mat -s 0 -n $neg -w -a $a
    python run.py $input result1/uniform_${dataset}_n${neg}_a${a}.mat -s 0 -n $neg -a $a
    python run_joint.py $input result1/joint_${dataset}_n${neg}_a${a}.mat -s 2 -m 1 -n $neg -a $a
  done
) &

export CUDA_VISIBLE_DEVICES="3"
neg=30
(
for a in 5 2 1 0.5 0.25
  do
    python run.py $input result1/uniform_${dataset}_nw_n${neg}_a${a}.mat -s 0 -n $neg -w -a $a
    python run.py $input result1/uniform_${dataset}_n${neg}_a${a}.mat -s 0 -n $neg -a $a
    python run_joint.py $input result1/joint_${dataset}_n${neg}_a${a}.mat -s 2 -m 1 -n $neg -a $a
  done
) &

wait
  
