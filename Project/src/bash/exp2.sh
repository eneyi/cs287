#!/bin/bash

# # preprocess all for joint version
# python preprocess.py --f all

# # 2 hop TE PE adjacent joint (2 times for robustness)
# # (estimated time 2 * 100 * 55 s = 3 hours-------------------)
# for i in `seq 1 2`;
# do
#     th mem2_pe.lua -filename all -hops 2 -adjacent 1 -nepochs 100 -pe 1 -extension $i
# done
# # 3 hop TE PE adjacent joint (2 times for robustness)
# # (estimated time 2 * 100 * 80 s = 4 hours 20min-------------------)
# for i in `seq 1 2`;
# do
#     th mem2_pe.lua -filename all -hops 3 -adjacent 1 -nepochs 100 -pe 1 -extension $i
# done

# ############# EXP TASK BY TASK

for i in `seq 1 20`;
do
    python preprocess.py --task $i --f task$i
    th mem2_pe.lua -filename task$i -hops 3 -adjacent 1 -nepochs 80 -pe 1 -extension $i
done