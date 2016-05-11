#!/bin/bash

# preprocess all for joint version
python preprocess.py --f all

# ######## Models with TE/adjacent/joint

# 1 hop TE adjacent joint
th mem2_pe.lua -filename all -hops 1 -adjacent 1 -nepochs 2
# 2 hop TE adjacent joint (2 times for robustness)
for i in `seq 1 2`;
do
    th mem2_pe.lua -filename all -hops 2 -adjacent 1 -nepochs 2 -extension $i
done
# # 3 hop TE adjacent joint (2 times for robustness)
# for i in `seq 1 2`;
# do
#     th mem2_pe.lua -filename all -hops 3 -adjacent 1 -nepochs 100 -extension $i
# done

# # ######## Models with TE/PE/adjacent/joint

# # 1 hop TE PE adjacent joint
# th mem2_pe.lua -filename all -hops 1 -adjacent 1 -nepochs 100 -pe 1
# # 2 hop TE PE adjacent joint (2 times for robustness)
# for i in `seq 1 2`;
# do
#     th mem2_pe.lua -filename all -hops 2 -adjacent 1 -nepochs 100 -pe 1 -extension $i
# done
# # 3 hop TE PE adjacent joint (2 times for robustness)
# for i in `seq 1 2`;
# do
#     th mem2_pe.lua -filename all -hops 3 -adjacent 1 -nepochs 100 -pe 1 -extension $i
# done

# # ######## Models with TE/PE/RNNlike/joint
# th mem2_pe.lua -filename all -hops 3 -adjacent 0 -nepochs 100 -pe 1

