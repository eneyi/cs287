require 'hdf5';
require 'nn';
require 'helper.lua';

cmd = torch.CmdLine()

-- Cmd Args
cmd:option('-N', 5, 'Ngram size for the input')
cmd:option('--embed', 16, 'Embedding size of characters')
cmd:option('--hid', 80, 'Hidden layer dimension')
cmd:option('--eta', 0.01, 'Learning rate')
cmd:option('--batch', 10, 'Batchsize')
cmd:option('--Ne', 20, 'Number of epochs')
cmd:option('-algo', 'greedy', 'Algorithm to use: either greedy or viterbi')
cmd:option('-f', 'pred_test.f5', 'File name for the predictions on the test')

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


-- Compute perplexity on entry with space
function compute_perplexity(gram_input, nnlm, N)
    local perp = 0
    local context = torch.zeros(N-1)
    local probability = torch.zeros(2)
    -- Do not predict for the last char
    --for i=1,gram_input:size(1)-N do
    local size=gram_input:size(1) - (N-1)
    for i=1,size do
        context:copy(gram_input:narrow(1,i,N-1))
        -- Line where the model appears
        probability:copy(nnlm:forward(context))
        if gram_input[i+(N-1)] == 1 then
            right_proba = probability[1]
        else
            right_proba = probability[2]
        end
        perp = perp + right_proba
    end
    perp = math.exp(-perp/size)
    return perp
end


-- Greedy algorithm to predict a sequence from gram_input with a count
-- based probability model
function predict_NN_greedy(gram_input, nnlm, N)
    -- Next Position to fill in predictions
    local position = N
    -- We allocate the maximum of memory that could be needed
    -- Default value is -1 (to know where predictions end afterwards)
    local predictions = torch.ones(2*(gram_input:size(1) - N)):mul(-1)
    -- Copy the first (N-1) gram
    predictions:narrow(1,1,N-1):copy(gram_input:narrow(1,1,N-1))
    local probability = torch.zeros(2)
    local context = torch.zeros(N-1)

    -- Build mapping
    for i=1,gram_input:size(1)-N do
        -- Compute proba for next char
        context:copy(predictions:narrow(1,position-(N-1),N-1))
        -- Line where the model appears
        probability:copy(nnlm:forward(context))
        m,a = probability:max(1)

        -- Case space predicted
        if (a[1] == 1) then
            predictions[position] = 1
            position = position +1
        end

        -- Copying next character
        predictions[position] = gram_input[i+N-1]
        position = position +1
    end
    -- Adding last character (</s>)
    predictions[position] = gram_input[gram_input:size(1)]
    -- Cutting the output
    return predictions:narrow(1,1,position)
end   

function predict_NN_greedy(gram_input, nnlm, N)
    -- Next Position to fill in predictions
    local position = N
    -- We allocate the maximum of memory that could be needed
    -- Default value is -1 (to know where predictions end afterwards)
    local predictions = torch.ones(2*(gram_input:size(1) - N)):mul(-1)
    -- Copy the first (N-1) gram
    predictions:narrow(1,1,N-1):copy(gram_input:narrow(1,1,N-1))
    local probability = torch.zeros(2)
    local context = torch.zeros(N-1)

    -- Build mapping
    for i=1,gram_input:size(1)-N do
        -- Compute proba for next char
        context:copy(predictions:narrow(1,position-(N-1),N-1))
        -- Line where the model appears
        probability:copy(nnlm:forward(context))
        m,a = probability:max(1)

        -- Case space predicted
        if (a[1] == 1) then
            predictions[position] = 1
            position = position +1
        end

        -- Copying next character
        predictions[position] = gram_input[i+N-1]
        position = position +1
    end
    -- Adding last character (</s>)
    predictions[position] = gram_input[gram_input:size(1)]
    -- Cutting the output
    return predictions:narrow(1,1,position)
end   

function predict_NN_greedy_cutoff(gram_input, nnlm, N, cut)
    -- Next Position to fill in predictions
    local position = N
    -- We allocate the maximum of memory that could be needed
    -- Default value is -1 (to know where predictions end afterwards)
    local predictions = torch.ones(2*(gram_input:size(1) - N)):mul(-1)
    -- Copy the first (N-1) gram
    predictions:narrow(1,1,N-1):copy(gram_input:narrow(1,1,N-1))
    local probability = torch.zeros(2)
    local context = torch.zeros(N-1)

    -- Build mapping
    for i=1,gram_input:size(1)-N do
        -- Compute proba for next char
        context:copy(predictions:narrow(1,position-(N-1),N-1))
        -- Line where the model appears
        probability:copy(nnlm:forward(context))
        -- Case space predicted
        if probability[1] > math.log(cut) then
            predictions[position] = 1
            position = position +1
        end

        -- Copying next character
        predictions[position] = gram_input[i+N-1]
        position = position +1
    end
    -- Adding last character (</s>)
    predictions[position] = gram_input[gram_input:size(1)]
    -- Cutting the output
    return predictions:narrow(1,1,position)
end   

-- Viterbi algorithm to predict a sequence from gram_input with a count
-- based probability model
-- pi matrix format (col1: space; col2: char)
function predict_NN_viterbi(gram_input, nnlm, N)
    -- Backpointer
    local score
    local bp = torch.zeros(gram_input:size(1) + 1, 2)
    local context = torch.DoubleTensor(1)
    local y_hat = torch.DoubleTensor(2)
    local pi = torch.ones(gram_input:size(1) + 1, 2):mul(-9999)
    -- Initialization
    pi[{1,1}] = 0
    -- i is shifted
    for i=2,gram_input:size(1)+1 do
        for c_prev =1,2 do
            -- Precompute y_hat(c_prev)
            if c_prev == 1 then
                context[1] = c_prev
            else
                context[1] = gram_input[i-1]
            end
            -- Line where the model appears
            y_hat:copy(nnlm:forward(context))

            for c_current =1,2 do
                score = pi[{i-1, c_prev}] + y_hat[c_current]
                if score > pi[{i, c_current}] then
                    pi[{i, c_current}] = score
                    bp[{i, c_current}] = c_prev
                end
            end
        end
    end
    return pi, bp
end

-- Building the sequences from the backpointer
function build_sequences_from_bp(bp, gram_input)
    local predictions = torch.DoubleTensor(2*gram_input:size(1))
    -- Next position to fill in predictions (have to do it backward)
    local position = 2*gram_input:size(1)
    local col = 2
    -- Loop until the 3rd position (because 2nd is the first one, could be set by hand)
    for i=bp:size(1),3,-1 do
        -- coming from a space
        if bp[i][col] == 1 then
            predictions[position] = 1
            position = position - 1
            col = 1
        else
            col = 2
        end
        -- index i is shifted of 1 wrt local index in gram_input
        predictions[position] = gram_input[i-1]
        position = position - 1
    end
    -- Beginnning of gram_input set
    predictions[position] = gram_input[1]
    position = position - 1

    return predictions:narrow(1,position+1,predictions:size(1)-position)
end

function main() 
    -- Parse input params
    opt = cmd:parse(arg)
    N = opt.N
    algo = opt.algo
    eta = opt.eta
    hid = opt.hid
    embed = opt.embed
    batchsize = opt.batch
    Ne = opt.Ne


    -- Reading file
    local file = hdf5.open('data_preprocessed/'..tostring(N)..'-grams.hdf5', 'r')
    data = file:all()
    file:close()

    train_input = data['input_matrix_train']
    train_output = data['output_matrix_train']
    input_data_train = data['input_data_train']

    input_data_valid = data['input_data_valid_nospace']:clone()

    input_data_test = data['input_data_test']:clone()
    
    -- Building the model
    torch.manualSeed(1)

    nnlm1, crit = build_model(N-1, 49, 2, embed, hid)

    print('-> Training the model')
    train_model(train_input, train_output, nnlm1, crit, N-1, 2, eta, Ne, batchsize)

    print('Ngram size '..tostring(N))
    print('Train Perplexity')
    print(compute_perplexity(input_data_train, nnlm1, N))
    print('Valid Perplexity')
    print(compute_perplexity(input_data_valid, nnlm1, N))

    -- Prediction
    if (algo == 'greedy') then
        predictions_test = predict_NN_greedy(input_data_test, nnlm1, N)
    elseif (algo == 'viterbi') then
        pi, bp = predict_count_based_viterbi(input_data_test, nnlm1, N)
        predictions_test = build_sequences_from_bp(bp, input_data_test)
    else
        error("invalid algorithm input") 
    end

    -- Kaggle format
    num_spaces = get_kaggle_format(predictions_test, N)

    print(num_spaces:narrow(1,1,10))

    -- -- Saving the Kaggle format output
    -- myFile = hdf5.open('submission/'..opt.f, 'w')
    -- myFile:write('num_spaces', num_spaces)
    -- myFile:close()
end

main()
