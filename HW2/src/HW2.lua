-- Only requirements allowed
require("hdf5")
require("nn")
require("optim")

cmd = torch.CmdLine()

-- Cmd Args
cmd:option('-datafile', 'PTB.hdf5', 'pre-processed data file')
cmd:option('-classifier', 'nb', 'classifier to use')
cmd:option('-f', '_pred_test.f5', 'File name for the predictions on the test')

-- Hyperparameters
-- The parameters are defined one by one, depending on the classifier call

-- Common hyperparameters for the gradient descent
cmd:option('-eta', '0.01', 'gradient descent learning rate parameter')
cmd:option('-batch_size', '20', 'size of the mini-batch for the stochastic gradient descent')
cmd:option('-ep_max', '5', 'number of epoch (i.e. updates of the gradient by record) for the stochastic gradient descent')

-- Hyperparameters for the Neural Network
cmd:option('-embedding', '1', 'boolean value to use the embeddings to initiliaze the lookup table (the dim hidden 1 parameter won\'t be considered if set to true')
cmd:option('-dim_hidden1', '50', 'Dimension of the first linear layer in the NN')
cmd:option('-dim_hidden2', '50', 'Dimension of the first linear layer in the NN')


-- Hyperparameters for the Naive Bayes
cmd:option('-alpha', '0.5', 'smoothing parameter')


--------------------------
-- Naive Bayes
--------------------------

function NaiveBayes(Xtrain_words, Xtrain_cap, Ytrain, nwords, ncap, nclasses, alpha)
    local n = Ytrain:size(1)
    local window_size = Xtrain_words:size(2)
    
    -- Building the prior
    local prior = torch.zeros(nclasses)
    for i = 1, n do
        c = Ytrain[i]
        prior[c] = prior[c] + 1
    end
    prior:div(n)

    -- Building the count matrix for words and caps with alpha as smoothing parameter
    
    local F_word = torch.zeros(nwords, nclasses)
    local F_cap = torch.zeros(ncap, nclasses)
    local x_word = torch.DoubleTensor(window_size)
    local x_cap = torch.DoubleTensor(window_size)
    -- Alpha smoothing parameter only use for the words features
    F_word:fill(alpha)
    for i = 1, n do
        c = Ytrain[i]
        x_word:copy(Xtrain_words[i])
        F_word:narrow(2,c,1):indexAdd(1, x_word:type('torch.LongTensor'), torch.ones(window_size,1))
        x_cap:copy(Xtrain_cap[i])
        F_cap:narrow(2,c,1):indexAdd(1, x_cap:type('torch.LongTensor'), torch.ones(window_size,1))
    end

    -- Building p(x|y) (same memory loc as F_word)
    local x_conditional_y_word = F_word
    -- Normalization
    x_conditional_y_word:cdiv(torch.expand(x_conditional_y_word:sum(1), nwords, nclasses))

    -- Building p(x|y) (same memory loc as F_cap)
    local x_conditional_y_cap = F_cap
    -- Normalization
    x_conditional_y_cap:cdiv(torch.expand(x_conditional_y_cap:sum(1), ncap, nclasses))
    
    return prior, x_conditional_y_word, x_conditional_y_cap
end

function predict_NB(Xword, Xcap, prior, x_conditional_y_word, x_conditional_y_cap, nclasses, Y)
    local n = Xword:size(1)
    local window_size = Xword:size(2)
    -- Building the conditional p(y|x)
    local y_conditional_x = torch.ones(n, nclasses)
    local x_word = torch.DoubleTensor(window_size)
    local x_cap = torch.DoubleTensor(window_size)
    for i = 1, n do
        x_word:copy(Xword[i])
        x_cap:copy(Xcap[i])
        for j=1, window_size do
            y_conditional_x[i]:cmul(x_conditional_y_word:narrow(1, x_word[j], 1))
            y_conditional_x[i]:cmul(x_conditional_y_cap:narrow(1, x_cap[j], 1))
        end
        -- Multiplying with the prior
        y_conditional_x[i]:cmul(prior)
    end
    -- Predicting
    max, Ypred = y_conditional_x:max(2)
    -- Computing accuracy if Y provided
    if Y then
        right = 0
        for i = 1, n do
            if Y[i] == Ypred[i][1] then
                right = right + 1
            end
        end
        accuracy = right / n
        print("Accuracy: " .. accuracy)
        return Ypred, accuracy
    else
        return Ypred
    end
end

--------------------------
-- Neural Network
--------------------------

function compute_accuracy(pred, true_)
    max,argmax = pred:max(2)
    acc = 0
    for i = 1, true_:size(1) do
        if argmax[i][1] == true_[i] then
            acc = acc + 1
        end
    end
    score = acc/true_:size(1)
    
    return score
end


-- If words_embeddings is nil, weight are initialized randomly by torch
function define_nn(nwords, ncap, nclasses, dim_hidden, dim_hidden2, word_embeddings)
    --Define the module
    neuralnet_wc = nn.Sequential()

    par = nn.LookupTable(nwords + ncap,dim_hidden)
    
    -- Adding the embeddings
    if word_embeddings then
        par.weight:narrow(1, 1, nwords):copy(word_embeddings:narrow(1, 1, nwords))
    end
    neuralnet_wc:add(par)

    neuralnet_wc:add(nn.View(1,-1,10*dim_hidden))
    neuralnet_wc:add(nn.Squeeze()) -- this layer is to go from a 1xAxB tensor to AxB dimensional tensor (https://groups.google.com/forum/#!topic/torch7/u4OEc0GB74k)
    neuralnet_wc:add(nn.Linear(10*dim_hidden,dim_hidden2))
    neuralnet_wc:add(nn.HardTanh())
    neuralnet_wc:add(nn.Linear(dim_hidden2, nclasses))
    neuralnet_wc:add(nn.LogSoftMax())
    
    return neuralnet_wc
end

function train_nn(train_output, train_input_word_windows, train_input_cap_windows, neuralnet_wc, ep_max, eta)
    train_new = torch.cat(train_input_word_windows,
        torch.add(train_input_cap_windows, 100002),2)
    -- Formating the dataset to train
    dataset={};
    for i=1,train_new:size(1) do 
      dataset[i] = {train_new[i]:view(1,10), train_output[i]}
    end
    function dataset:size() return train_new:size(1) end -- 100 examples

    criterion = nn.ClassNLLCriterion()

    -- Training
    timer = torch.Timer()
    trainer = nn.StochasticGradient(neuralnet_wc, criterion)
    trainer.learningRate = eta
    trainer.maxIteration = ep_max

    trainer:train(dataset)

    return neuralnet_wc
end

function pred_nn(input_word_windows, input_cap_windows, neuralnet_wc, output)
    new = torch.cat(input_word_windows,
        torch.add(input_cap_windows, 100002),2)
    pred_val = neuralnet_wc:forward(new)
    if output then
        valid_accuracy = compute_accuracy(pred_val, output)
        return pred_val, accuracy
    end
    return pred_val
end

function main() 
    -- Parse input params
    opt = cmd:parse(arg)
    local f = hdf5.open(opt.datafile, 'r')
    train_input_word_windows = f:read('train_input_word_windows'):all()
    train_input_cap_windows = f:read('train_input_cap_windows'):all()
    valid_input_word_windows = f:read('valid_input_word_windows'):all()
    valid_input_cap_windows = f:read('valid_input_cap_windows'):all()
    train_output = f:read('train_output'):all()
    valid_output = f:read('valid_output'):all()
    word_embeddings = f:read('word_embeddings'):all()
    nclasses = f:read('nclasses'):all()[1]
    nwords = f:read('nwords'):all()[1]
    ncap = 4

    -- Reading the model
    classifier = opt.classifier

    if ((classifier == 'nb') || (classifier == 'log_reg')) then
        -- Process the features for NB and log reg
        for j = 1, 5 do
            train_word:narrow(2,j,1):add((j-1)*100002)
            valid_word:narrow(2,j,1):add((j-1)*100002)
        end
        for j = 1, 5 do
            train_cap:narrow(2,j,1):add((j-1)*4)
            valid_cap:narrow(2,j,1):add((j-1)*4)
        end
        for j = 1, 5 do
            test_word:narrow(2,j,1):add((j-1)*4)
            test_cap:narrow(2,j,1):add((j-1)*4)
        end

        if (classifier == 'nb') then
            -- Hyperparameters
            alpha = opt.alpha

            timer = torch.Timer()
            -- Learning the model
            prior, x_conditional_y_word, x_conditional_y_cap = NaiveBayes(train_word, train_cap, train_output, nwords_bis, ncap_bis, nclasses, alpha)
            -- Prediction on the validation set
            Ypred_validate, accuracy_validate = predict_NB(valid_word, valid_cap, prior, x_conditional_y_word, x_conditional_y_cap, nclasses, valid_output)
            -- Prediction on the test set
            Ypred_validate, accuracy_validate = predict_NB(test_word, test_cap, prior, x_conditional_y_word, x_conditional_y_cap, nclasses)

            print('Time elapsed for NB ' .. timer:time().real .. ' seconds',"\n")
        end
    elseif (classifier == 'nn') then
        dim_hidden1 = opt.dim_hidden1
        dim_hidden2 = opt.dim_hidden2
        if opt.embeddings then
            neuralnet_wc = define_nn(nwords, ncap, nclasses, dim_hidden1, dim_hidden2, word_embeddings)
        else
            neuralnet_wc = define_nn(nwords, ncap, nclasses, dim_hidden1, dim_hidden2, word_embeddings)
        end
    end

    -- Saving the predictions on test
    filename = classifier .. opt.f
    if (filename) then
        myFile = hdf5.open(filename, 'w')
        myFile:write('Testpred', Testpred)
        myFile:close()
    end
end

main()
