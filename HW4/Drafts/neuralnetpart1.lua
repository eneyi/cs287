require 'hdf5'
require 'nn'

function build_model(dwin, nchar, nclass, hid1, hid2)
    -- Model with skip layer from Bengio, standards parameters
    -- should be:
    -- dwin = 5
    -- hid1 = 30
    -- hid2 = 100

    -- To store the whole model
    local dnnlm = nn.Sequential()

    -- Layer to embedd (and put the words along the window into one vector)
    local LT = nn.Sequential()
    local LT_ = nn.LookupTable(nchar,hid1)
    LT:add(LT_)
    LT:add(nn.View(-1, hid1*dwin))

    dnnlm:add(LT)

    local concat = nn.ConcatTable()

    local lin_tanh = nn.Sequential()
    lin_tanh:add(nn.Linear(hid1*dwin,hid2))
    lin_tanh:add(nn.Tanh())

    local id = nn.Identity()

    concat:add(lin_tanh)
    concat:add(id)

    dnnlm:add(concat)
    dnnlm:add(nn.JoinTable(2))
    dnnlm:add(nn.Linear(hid1*dwin + hid2, nclass))
    dnnlm:add(nn.LogSoftMax())

    -- Loss
    local criterion = nn.ClassNLLCriterion()

    return dnnlm, criterion
end


function train_model(train_input, train_output, dnnlm, criterion, dwin, nclass, eta, nEpochs, batchSize)
    -- Train the model with a mini batch SGD
    -- standard parameters are
    -- nEpochs = 1
    -- batchSize = 32
    -- eta = 0.01

    -- To store the loss
    local av_L = 0

    -- Memory allocation
    local inputs_batch = torch.DoubleTensor(batchSize,dwin)
    local targets_batch = torch.DoubleTensor(batchSize)
    local outputs = torch.DoubleTensor(batchSize, nclass)
    local df_do = torch.DoubleTensor(batchSize, nclass)

    for i = 1, nEpochs do
        -- timing the epoch
        local timer = torch.Timer()

        av_L = 0
        
        -- max renorm of the lookup table
        dnnlm:get(1):get(1).weight:renorm(2,1,1)
        
        -- mini batch loop
        for t = 1, train_input:size(1), batchSize do
            -- Mini batch data
            local current_batch_size = math.min(batchSize,train_input:size(1)-t)
            inputs_batch:narrow(1,1,current_batch_size):copy(train_input:narrow(1,t,current_batch_size))
            targets_batch:narrow(1,1,current_batch_size):copy(train_output:narrow(1,t,current_batch_size))
            
            -- reset gradients
            dnnlm:zeroGradParameters()
            --gradParameters:zero()

            -- Forward pass (selection of inputs_batch in case the batch is not full, ie last batch)
            outputs:narrow(1,1,current_batch_size):copy(dnnlm:forward(inputs_batch:narrow(1,1,current_batch_size)))

            -- Average loss computation
            local f = criterion:forward(outputs:narrow(1,1,current_batch_size), targets_batch:narrow(1,1,current_batch_size))
            av_L = av_L +f

            -- Backward pass
            df_do:narrow(1,1,current_batch_size):copy(criterion:backward(outputs:narrow(1,1,current_batch_size), targets_batch:narrow(1,1,current_batch_size)))
            dnnlm:backward(inputs_batch:narrow(1,1,current_batch_size), df_do:narrow(1,1,current_batch_size))
            dnnlm:updateParameters(eta)
            
        end
            
        print('Epoch '..i..': '..timer:time().real)
        print('Average Loss: '..av_L/math.floor(train_input:size(1)/batchSize))
       
    end

end








