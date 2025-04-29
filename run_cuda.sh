#!/bin/bash
#SBATCH --job-name=GPU
#SBATCH -N 1 # number of nodes
#SBATCH -n 1 # number of cores
## From PC labs: cuda-int; otherwise, use cuda-ext.q
# For professors: cuda-staff
#SBATCH --partition=cuda-ext.q
## GPUs are available on aolin cluster
#SBATCH --gres=gpu:GeForceRTX3080:1
#SBATCH -o out.out
#SBATCH -e err.out

set -e
module load nvhpc/21.2

compile(){
  mkdir -p build && cd build
  gcc -Ofast -c ../common/common.c ../configuration/config.c ../layer/layer.c ../randomizer/randomizer.c ../initialize/initialize.c ../training/training.c
  nvcc -c ../main.cu -o main.o
  nvcc main.o common.o config.o layer.o randomizer.o initialize.o training.o -o ../exec && cd ../
}
compile
./exec

# Profiling:
#nsys nvprof --print-gpu-trace ./exec #summary 
#ncu --target-processes application-only --set full -f -o profile.ncu-rep ./exec

#Visualize profiling:
# 1. Manually download file "profile.ncu-rep" to your computer
# 2. Download the "Nsys" profiler from the Nvidia website on your computer (https://developer.nvidia.com/nsight-systems).
# 3. Use "ncu" from your computer to open your profile report
# Profile traces with nsys 
#nsys profile -f true -t nvtx,cuda -o profile.nsys-rep ./exec
# Download "profile.nsys-rep.qdrep" on your computer and use nsys to open the file.
