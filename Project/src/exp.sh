#!/bin/bash

# preprocess all for joint version
python preprocess.py --f all

# ######## Models with TE/adjacent/joint

# 1 hop TE adjacent joint 
# (estimated time 100 * 20 s = 35min-------------------)
th mem2_pe.lua -filename all -hops 1 -adjacent 1 -nepochs 100
# 2 hop TE adjacent joint (2 times for robustness)
# (estimated time 2 * 100 * 35 s = 2 hours-------------------)
for i in `seq 1 2`;
do
    th mem2_pe.lua -filename all -hops 2 -adjacent 1 -nepochs 100 -extension $i
done
# 3 hop TE adjacent joint (2 times for robustness)
# (estimated time 2 * 100 * 47 s = 2 hours 40 min-------------------)
for i in `seq 1 2`;
do
    th mem2_pe.lua -filename all -hops 3 -adjacent 1 -nepochs 100 -extension $i
done
# ######## Models with TE/PE/adjacent/joint
# 1 hop TE PE adjacent joint
# (estimated time 100 * 30 s = 50min-------------------)
th mem2_pe.lua -filename all -hops 1 -adjacent 1 -nepochs 100 -pe 1
# ######## Models with TE/PE/RNNlike/joint
# (estimated time 100 * 40 s = 70min-------------------)
th mem2_pe.lua -filename all -hops 3 -adjacent 0 -nepochs 100 -pe 1