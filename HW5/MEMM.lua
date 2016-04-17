require 'hdf5';
require 'nn';
require 'helper.lua';

-- Loading data
myFile = hdf5.open('../data/MM_data_pos.hdf5','r')
data = myFile:all()
input_matrix_train_pos = data['input_matrix_train_pos']
input_matrix_dev_pos = data['input_matrix_dev_pos']
input_matrix_test_pos = data['input_matrix_test_pos']
myFile:close()

nwords = input_matrix_train_pos:size(1)
train_output = input_matrix_train_pos:narrow(2,59)
train_input_pos = torch.Tensor(nwords-1,1+9+5+43)
train_input_pos:narrow(2,1,1):copy(input_matrix_train_pos:narrow(2,1,1):narrow(1,2,nwords-1))
train_input_pos:narrow(2,2,9):copy(input_matrix_train_pos:narrow(2,2,9):narrow(1,1,nwords-1))
train_input_pos:narrow(2,11,5):copy(input_matrix_train_pos:narrow(2,11,5):narrow(1,1,nwords-1))
train_input_pos:narrow(2,16,43):copy(input_matrix_train_pos:narrow(2,16,43):narrow(1,1,nwords-1))

observations_dev = input_matrix_dev_pos:narrow(2,1,1):clone()
dev_feat = input_matrix_dev_pos:narrow(2,11, 5 + 43)
dev_true_classes = input_matrix_dev_pos:narrow(2, 59,1):squeeze()

observations_test_pos = input_matrix_test_pos:narrow(2,1,1)
observations_test_feat = input_matrix_test_pos:narrow(2,2,5+43)

-- Defining the model

model = nn.Sequential()
t1_pos = nn.ParallelTable()

t1_pos_1 = nn.Sequential()
t1_pos_1:add(LT)
t1_pos_1:add(nn.View(-1,50))

t1_pos_2 = nn.Identity()

t1_pos:add(t1_pos_1)
t1_pos:add(t1_pos_2)

model:add(t1_pos)
model:add(nn.JoinTable(2))

model:add(nn.Linear(50 + 9 + 5 + 43,9))
model:add(nn.LogSoftMax())

-- Training function:


function train_model_cap(train_input, train_output, model, criterion, din, nclass, eta, nEpochs, batchSize)
    -- Train the model with a mini batch SGD
    -- standard parameters are
    -- nEpochs = 1
    -- batchSize = 32
    -- eta = 0.01
    local loss = torch.Tensor(nEpochs)

    -- To store the loss
    local av_L = 0

    -- Memory allocation
    local inputs_batch = torch.DoubleTensor(batchSize, din)
    local targets_batch = torch.DoubleTensor(batchSize)
    local outputs = torch.DoubleTensor(batchSize, nclass)
    local  df_do = torch.DoubleTensor(batchSize, nclass)

    for i = 1, nEpochs do
        -- timing the epoch
        timer = torch.Timer()
        av_L = 0
        
        -- mini batch loop
        for t = 1, train_input:size(1), batchSize do
            -- Mini batch data
            current_batch_size = math.min(batchSize,train_input:size(1)-t)

            inputs_batch:narrow(1,1,current_batch_size):copy(train_input:narrow(1,t,current_batch_size))
            
            targets_batch:narrow(1,1,current_batch_size):copy(train_output:narrow(1,t,current_batch_size))
            
            -- reset gradients
            model:zeroGradParameters()

            -- Forward pass (selection of inputs_batch in case the batch is not full, ie last batch)
            outputs:narrow(1,1,current_batch_size):copy(model:forward({inputs_batch:narrow(1,1,current_batch_size):narrow(2,1,1),
            inputs_batch:narrow(1,1,current_batch_size):narrow(2,2,din-1)}))
            -- Average loss computation
            f = criterion:forward(outputs:narrow(1,1,current_batch_size), targets_batch:narrow(1,1,current_batch_size))
            
            av_L = av_L +f

            -- Backward pass
            df_do:narrow(1,1,current_batch_size):copy(criterion:backward(outputs:narrow(1,1,current_batch_size), targets_batch:narrow(1,1,current_batch_size)))
            model:backward({inputs_batch:narrow(1,1,current_batch_size):narrow(2,1,1), inputs_batch:narrow(1,1,current_batch_size):narrow(2,2,din-1)}, 
            df_do:narrow(1,1,current_batch_size))

            model:updateParameters(eta)
            
        end
            
        print('Epoch '..i..': '..timer:time().real)
        loss[i] = av_L/math.floor(train_input:size(1)/batchSize)
        print('Average Loss: '.. loss[i])
       
    end
    
    return loss
end

-- Viterbi for MEMM:

-- Evaluates the matrix of scores for all possible  tags for the previous word, using the word features at timestep i

function compute_logscore_extrafeat(observations, feat, i, model, C)
    local y = torch.zeros(C,C)
    local hot_1 = torch.zeros(C+feat:size(2))
    for j = 1, C do
        hot_1:zero()
        hot_1[j] = 1
        hot_1:narrow(1,10,feat:size(2)):copy(feat:narrow(1,i,1))
        y:narrow(1,j,1):copy(model:forward({observations[i]:view(1,1),hot_1:view(1,C+feat:size(2))}))
    end
    return y
end

-- Evaluates the highest scoring sequence:
function viterbi_extrafeat(observations, feat, compute_logscore, model, C)
    
    local y = torch.zeros(C,C)
    -- Formating tensors
    local initial = torch.zeros(C, 1)
    -- initial started with a start of sentence: <t>

    initial[{8,1}] = 1
    initial:log()

    -- number of classes
    local n = observations:size(1)
    local max_table = torch.Tensor(n, C)
    local backpointer_table = torch.Tensor(n, C)
    -- first timestep
    -- the initial most likely paths are the initial state distribution
    local maxes, backpointers = (initial + compute_logscore_extrafeat(observations, feat, 1, model, C)[8]):max(2)
    max_table[1] = maxes
    -- remaining timesteps ("forwarding" the maxes)
    for i=2,n do
        -- precompute edge scores
       
        y:copy(compute_logscore_extrafeat(observations, feat, i, model, C))
        scores = y:transpose(1,2) + maxes:view(1, C):expand(C, C)

        -- compute new maxes 
        maxes, backpointers = scores:max(2)

        -- record
        max_table[i] = maxes
        backpointer_table[i] = backpointers
    end
    -- follow backpointers to recover max path
    local classes = torch.Tensor(n)
    maxes, classes[n] = maxes:max(1)
    for i=n,2,-1 do
        classes[i-1] = backpointer_table[{i, classes[i]}]
    end

    return classes
end

-- Train Model

loss_pos = train_model_cap(train_input_pos, train_output, ultimate_t_pos, criterion, 1 + 9 + 5 + 43, 9, 0.1, 20, 32)

-- Evaluate performance on dev set:

cl_pos_dev = viterbi_extrafeat(observations_dev, dev_feat, compute_logscore_extrafeat, ultimate_t_pos, 9)
f = f_score(cl_pos_dev, dev_true_classes)

-- Predict on test:

v_seq_test_pos = viterbi_extrafeat(observations_test_pos, observations_test_feat, compute_logscore_extrafeat, ultimate_t_pos, 9)

-- Saving predicted sequence on test
myFile = hdf5.open('../submission/v_seq_test_mem_pos', 'w')
myFile:write('v_seq_test', v_seq_test_pos)
myFile:close()
