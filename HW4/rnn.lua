-- Documentation:
-- ---- How to call it from the command line?
-- For example:
-- $ th count_based.lua -N 5
-- Other argument possible (see below)
-- 
-- ---- Is there an Output?
-- By default, the predictions on the test set are saved in hdf5 format as classifier .. opt.f

-- Only requirements allowed
require("hdf5")
require("rnn")
require 'helper.lua';

cmd = torch.CmdLine()

-- Cmd Args
cmd:option('-l', 30, 'Length size for the training sequence')
cmd:option('-b', 16, 'Batch-size for the training')
cmd:option('-edim', 20, 'Embed dimension for the characters embeddings')
cmd:option('-eta', 0.5, 'Learning rate')
cmd:option('-ne', 4, 'Number of epochs for the training')
cmd:option('-s', 1, 'Step size for the adaptive eta changes')
cmd:option('-f', 'pred_test_rnn.f5', 'File name for the predictions on the test')
cmd:option('-model', 'RNN', 'Recurrent model to be used (RNN, LSTM or GRU')


-- Formating the input
-- input is a 1d tensor
function get_train_input(input, len, batch_size)
    -- Building output (we put predict a padding at the end)
    local n = input:size(1)
    
    -- Get the closer multiple of batch_size*len below n
    local factor = -math.floor(-n/(len*batch_size))
    local n_new = factor*len*batch_size
    local input_new = torch.DoubleTensor(n_new)
    local t_input, t_output
    input_new:narrow(1,1,n):copy(input)
    input_new:narrow(1,n,n_new-n+1):fill(2) -- Filling with padding
    
    -- Building output
    local output = get_output(input_new)

    -- Issue with last sequence if batch_size does not divide n
    t_input = torch.split(input_new:view(batch_size,n_new/batch_size),len, 2)
    t_output = torch.split(output:view(batch_size,n_new/batch_size),len, 2)
    return t_input, t_output
end 

function get_output(input)
    local n = input:size(1)
    local output = torch.DoubleTensor(n)
    for i=2, n do
        if input_new[i] ~= 1 then
            output[i-1] = 2
        else
            output[i-1] = input[i]
        end
    end
    output[n] = 2
    return output
end

-- Methods to build the model
function build_RNN(embed_dim, rho)
    return nn.Recurrent(embed_dim, nn.Linear(embed_dim, embed_dim),nn.Linear(embed_dim, embed_dim), nn.Tanh(), rho)
end

function build_LSTM(embed_dim, rho)
    return nn.FastLSTM(embed_dim, embed_dim, rho)
end

function build_GRU(embed_dim, rho, dropout_p)
    return nn.GRU(embed_dim, embed_dim, rho,dropout_p)
end

function build_rnn(embed_dim, vocab_size, batch_size, recurrent_model, len)
    local batchRNN
    local params
    local grad_params
    -- generic RNN transduced
    batchRNN = nn.Sequential()
        :add(nn.LookupTable(vocab_size, embed_dim))
        :add(nn.SplitTable(1, batch_size))
    local rec = nn.Sequencer(recurrent_model)
    rec:remember('both')
    
    batchRNN:add(rec)
    
    -- Output
    batchRNN:add(nn.Sequencer(nn.Linear(embed_dim, 2)))
    batchRNN:add(nn.Sequencer(nn.LogSoftMax()))

    -- Retrieve parameters (To do only once!!!)
    params, grad_params = batchRNN:getParameters()
    -- Initializing all the parameters between -0.05 and 0.05
    for k=1,params:size(1) do
        params[k] = torch.uniform(-0.05,0.05)
    end
    
    return batchRNN, params, grad_params
end

function train_model_with_perp(t_input, t_output, model, model_flattened, params_flattened,
        params, grad_params, criterion, eta, nEpochs, batch_size, len, n, input_valid, output_valid, step)
    -- Train the model with a mini batch SGD
    -- Uses an adaptive learning rate eta computed each cycle of step iterations from the
    -- evolution of the perplexity on the validation set (compute with the model_flattened)
    local timer
    local pred
    local loss
    local dLdPred
    local t_inputT = torch.DoubleTensor(len,batch_size)
    local t_output_table
    local size

    -- To store the loss
    local av_L = 0
    local perp = 0
    local old_perp = 0

    for i = 1, nEpochs do
        -- timing the epoch
        timer = torch.Timer()
        old_L = av_L
        old_perp = perp
        av_L = 0
        
        -- mini batch loop
        for k = 1, n/(batch_size * len) do
            -- Mini batch data
                
            t_inputT:copy(t_input[k]:t())
            t_output_table = torch.split(t_output[k],1,2)
            --format the output
            for j=1,len do
                t_output_table[j] = t_output_table[j]:squeeze()
            end 
            
            -- reset gradients
            grad_params:zero()
            
            -- Forward loop
            pred = model:forward(t_inputT)
            loss = criterion:forward(pred, t_output_table)
            av_L = av_L + loss

            -- Backward loop
            dLdPred = criterion:backward(pred, t_output_table)
            model:backward(t_inputT, dLdPred)
            
            -- gradient normalization with max norm 5 (l2 norm)
            grad_params:view(grad_params:size(1),1):renorm(1,2,5)
            model:updateParameters(eta)
            
        end
            
        print('Epoch '..i..': '..timer:time().real)
        print('Average Loss: '..av_L/math.floor(n/batch_size))
        -- Print perplexity validity every step of iteration
        if (i%step == 0) then
            size = input_valid:size(1) - 1
            params_flattened:copy(params)
            perp = compute_perplexity(input_valid:narrow(1,1,size):view(size,1), output_valid, model_flattened)
            print('Valid perplexity: '..perp)
            
            if old_perp - perp < 0 then
                eta = eta/2
            end

            if (eta < 0.0001) then eta = 0.1 end

        end
    end
end

---------------------------
----- Methods for prediction
---------------------------

function compute_probability_model(model, input)
    return model:forward(input:view(input:size(1), 1))
end

-- Method to compute manually the perplexity
function compute_perplexity(input, output, model)
    -- Last Position filled in predictions
    -- Position to predict in input
    local position_input = 1
    local probability = torch.DoubleTensor(2)
    local probability_table
    local perp = 0

    -- Build mapping
    for i = 1,input:size(1) do
        -- Line where the model appears
        -- The model remember the states before, just need to feed into it a character
        probability_table = compute_probability_model(model, input:narrow(1,i,1))
        probability:copy(probability_table[1])
        perp = perp + probability[output[i]]
    end
    -- Cutting the output
    return math.exp(-perp/input:size(1))
end   

-- Prediction with greedy algorithm
function predict_rnn_greedy(input, len, model)
    -- Last Position filled in predictions
    local position_prediction = 1
    -- Position to predict in input
    local position_input = 1
    -- We allocate the maximum of memory that could be needed
    -- Default value is -1 (to know where predictions end afterwards)
    local predictions = torch.ones(2*input:size(1)):mul(-1)
    -- Copy the first entry
    predictions[position_prediction] = input[position_input]
    local probability = torch.zeros(2)
    local probability_table

    -- Build mapping
    while position_input < input:size(1) do
        -- Line where the model appears
        -- The model remember the states before, just need to feed into it a character
        probability_table = compute_probability_model(model, predictions:narrow(1,position_prediction, 1))
        probability:copy(probability_table[1])

        m,a = probability:max(1)

        -- Case space predicted
        position_prediction = position_prediction +1
        if (a[1] == 1) then
            predictions[position_prediction] = 1
        else
            -- Copying next character
            position_input = position_input + 1
            predictions[position_prediction] = input[position_input] 
        end
    end
    -- Cutting the output
    return predictions:narrow(1,1,position_prediction)
end   

function main() 
    -- Parse input params
    opt = cmd:parse(arg)

    -- Reading file
    N = 2
    local data = hdf5.open('../data_preprocessed/'..tostring(N)..'-grams.hdf5','r'):all()
    F_train = data['F_train']
    input_data_valid = data['input_data_valid']
    input_matrix_train = data['input_matrix_train']
    input_data_train = data['input_data_train']
    input_data_valid_nospace = data['input_data_valid_nospace']
    input_data_test = data['input_data_test']
    myFile:close()

    F_train = data['F_train']
    input_data_valid = data['input_data_valid']
    input_data_train = data['input_data_train']
    input_data_test = data['input_data_test']
    input_data_valid_nospace = data['input_data_valid_nospace']

    -- Model parameters
    len = opt.l
    batch_size = opt.b
    vocab_size = 49
    embed_dim = oopt.edim
    eta = opt.eta
    nEpochs = opt.ne
    step = opt.s

    -- Formating data
    t_input_new, t_output_new = get_train_input(input_data_train, len, batch_size)
    output_valid = get_output(input_data_valid)
    n_new = len * batch_size *(#t_input_new)

    -- Building model
    model, params, grad_params = build_rnn(embed_dim, vocab_size, batch_size, build_RNN(embed_dim, len), len)
    model_valid, params_valid, grad_params_valid = build_rnn(embed_dim, vocab_size, 1,build_RNN(embed_dim))

    crit = nn.SequencerCriterion(nn.ClassNLLCriterion())

    -- Training model
    train_model_with_perp(t_input_new, t_output_new, model, model_valid, params_valid,
            params, grad_params, crit, eta, nEpochs, batch_size, len, n_new, input_data_valid, output_valid, step)
    print('here')

    -- -- Computing RMSE on valid
    -- kaggle_true_valid = get_kaggle_format(input_data_valid,2)

    -- timer = torch.Timer()
    -- pred_valid = predict_rnn_greedy(input_data_valid_nospace:narrow(1,1,input_data_valid_nospace:size(1)), len, model_valid)
    -- print('Greedy prediction on validation set (Time elasped : '..timer:time().real..' )')
    -- kaggle_model_valid = get_kaggle_format(pred_valid,2)
    -- print('RMSE')
    -- rsme = compute_rmse(kaggle_true_valid, kaggle_model_valid)
    -- print(rsme)

    -- -- Prediction on test
    -- timer = torch.Timer()
    -- size = input_data_test:size(1)
    -- pred_test = predict_rnn_greedy(input_data_test:narrow(1,1,size), len, model_valid)
    -- print('Greedy prediction on test set (Time elasped : '..timer:time().real..' )')
    -- kaggle_test = get_kaggle_format(pred_test,2)

    -- -- Saving the Kaggle format output
    -- myFile = hdf5.open('../submission/'..opt.f, 'w')
    -- myFile:write('num_spaces', kaggle_test)
    -- myFile:close()
end