function build_model(dwin, nwords, hid1, hid2)
    -- Model with skip layer from Bengio, standards parameters
    -- should be:
    -- dwin = 5
    -- hid1 = 30
    -- hid2 = 100

    -- To store the whole model
    dnnlm = nn.Sequential()

    -- Layer to embedd (and put the words along the window into one vector)
    LT = nn.Sequential()
    LT_ = nn.LookupTable(nwords,hid1)
    LT:add(LT_)
    LT:add(nn.View(-1, hid1*dwin))

    dnnlm:add(LT)

    concat = nn.ConcatTable()

    lin_tanh = nn.Sequential()
    lin_tanh:add(nn.Linear(hid1*dwin,hid2))
    lin_tanh:add(nn.Tanh())

    id = nn.Identity()

    concat:add(lin_tanh)
    concat:add(id)

    dnnlm:add(concat)
    dnnlm:add(nn.JoinTable(2))
    dnnlm:add(nn.Linear(hid1*dwin + hid2, nwords))
    dnnlm:add(nn.LogSoftMax())

    -- Loss
    criterion = nn.ClassNLLCriterion()

    return dnnlm, criterion
end


function train_model(train_input, dnnlm, criterion, dwin, nwords, eta, nEpochs, batchSize)
    -- Train the model with a mini batch SGD
    -- standard parameters are
    -- nEpochs = 1
    -- batchSize = 32
    -- eta = 0.01

    -- To store the loss
    av_L = 0

    -- Memory allocation
    inputs_batch = torch.DoubleTensor(batchSize,dwin)
    targets_batch = torch.DoubleTensor(batchSize)
    outputs = torch.DoubleTensor(batchSize, nwords)
    df_do = torch.DoubleTensor(batchSize, nwords)

    for i = 1, nEpochs do
        -- timing the epoch
        timer = torch.Timer()
        av_L = 0
        
        -- max renorm of the lookup table
        dnnlm:get(1):get(1).weight:renorm(2,1,1)
        
        -- mini batch loop
        for t = 1, train_input:size(1), batchSize do
            -- Mini batch data
            current_batch_size = math.min(batchSize,train_input:size(1)-t)
            inputs_batch:narrow(1,1,current_batch_size):copy(train_input:narrow(1,t,current_batch_size))
            targets_batch:narrow(1,1,current_batch_size):copy(train_output:narrow(1,t,current_batch_size))
            
            -- reset gradients
            dnnlm:zeroGradParameters()
            --gradParameters:zero()

            -- Forward pass (selection of inputs_batch in case the batch is not full, ie last batch)
            outputs:narrow(1,1,current_batch_size):copy(dnnlm:forward(inputs_batch:narrow(1,1,current_batch_size)))

            -- Average loss computation
            f = criterion:forward(outputs:narrow(1,1,current_batch_size), targets_batch:narrow(1,1,current_batch_size))
            av_L = av_L +f

            -- Backward pass
            df_do:narrow(1,1,current_batch_size):copy(criterion:backward(outputs:narrow(1,1,current_batch_size), targets_batch:narrow(1,1,current_batch_size)))
            dnnlm:backward(inputs_batch:narrow(1,1,current_batch_size), df_do:narrow(1,1,current_batch_size))
            dnnlm:updateParameters(eta)
            
        end
            
        print('Epoch '..i..': '..timer:time().real)
        print('Average Loss: '..av_L/math.floor(train_input:size(1)/batchSize))
       
    end

    return dnnlm

end