-- Documentation:
-- ---- How to call it from the command line?
-- For example:
-- $ th HW1.lua -datafile SST1.hdf5 -classifier linear_svm
-- Other argument possible (see below)
-- 
-- ---- Is there an Output?
-- By default, the predictions on the test set are saved in hdf5 format as classifier .. opt.f

-- Only requirement allowed
require("hdf5")

cmd = torch.CmdLine()

-- Cmd Args
cmd:option('-datafile', '', 'data file')
cmd:option('-classifier', 'nb', 'classifier to use')
cmd:option('-f', '_pred_test.f5', 'File name for the predictions on the test')

-- Hyperparameters
-- The parameters are defined one by one, depending on the classifier call

-- Common hyperparameters for hinge loss and logistic regression
cmd:option('-lambda', '1', 'regularization parameter')
cmd:option('-eta', '1', 'gradient descent learning rate parameter')
cmd:option('-batch_size', '20', 'size of the mini-batch for the stochastic gradient descent')
cmd:option('-ep_max', '1', 'number of epoch (i.e. updates of the gradient by record) for the stochastic gradient descent')

-- Hyperparameters for the Naive Bayes
cmd:option('-alpha', '1', 'smoothing parameter')

--------------------------
-- Naive Bayes training
--------------------------
function NaiveBayes(Xtrain, Ytrain, nfeatures, nclasses, alpha)
    timer = torch.Timer()
    
    local n = Xtrain:size(1)
    
    -- Building the prior
    
    local prior = torch.zeros(nclasses)
    for i = 1, n do
        c = train_output[i]
        prior[c] = prior[c] + 1
    end
    prior:div(n)
    
    -- Building the count matrix with alpha as smoothing parameter
    
    local F = torch.zeros(nfeatures + 1, nclasses)
    local x = torch.DoubleTensor(53)
    F:fill(alpha)
    for i = 1, n do
        c = train_output[i]
        x:copy(train_input[i])
        F:narrow(2,c,1):indexAdd(1, x:type('torch.LongTensor'), torch.ones(53,1))
    end
    F[1]:zero()
    
    -- Building p(x|y) (same memory loc as F)
    
    local x_conditional_y = F
    x_conditional_y:cdiv(torch.expand(x_conditional_y:sum(1), nfeatures + 1, nclasses))
    x_conditional_y[1]:zero()
    
    print('Time elapsed for Naive Bayes training : ' .. timer:time().real .. ' seconds')
    return prior, x_conditional_y
end

-- Generic function to predict the output of a linear model: y = Wx + b
-- Computes the accuracy if the Y_output is provided
-- A specific function follows for the Naive Bayes

function predict(Xvalid, W, b, Yvalid)
    local Yvalid = Yvalid or Nil
    local z = torch.zeros(5, 1)
    local Ypred = torch.IntTensor(Xvalid:size()[1])
    local max = torch.zeros(1)
    local accu = 0
    local n = Xvalid:size(1)
    
    for i = 1,n do
        z:copy(W:index(1,Xvalid[i]:type('torch.LongTensor')):sum(1):add(b)):view(-1,1)
        max, Ypred[i] = z:max(1)
        
        if Yvalid then
            if Yvalid[i] == Ypred[i] then
                accu = accu + 1
            end
        end
    end
    
    if Yvalid then
        print("Accuracy: " .. accu/n)
        return Ypred,accu/n
    else
        return Ypred
    end
    
end

function predict_NB(X, prior, x_conditional_y, nclasses, Y)   
    local n = X:size(1)
    
    -- Building the conditional p(y|x)
    
    local y_conditional_x = torch.ones(n, nclasses)
    local x = torch.DoubleTensor(53)
    for i = 1, n do
        x:copy(X[i])
        for j=1, X:size(2) do
            if x[j] ~= 1 then
                y_conditional_x[i]:cmul(x_conditional_y:narrow(1, x[j], 1))
            else
                break
            end
        end
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
-- Logistic Regression training
--------------------------
function logreg(Xtrain, Ytrain, nfeatures, nclasses, ep_max, batch_size, eta, lambda)
    
    -- Initialization
    local shuffle = torch.LongTensor(Xtrain:size(1))
    local W = torch.cat(torch.zeros(1,nclasses),torch.ones(nfeatures, nclasses),1)
    local b = torch.ones(nclasses, 1)
    local grad_W = torch.zeros(nfeatures+1,nclasses)
    local grad_b = torch.zeros(nclasses, 1)
    local z = torch.ones(nclasses, 1)
    local yhat = torch.ones(nclasses, 1)
    local grad_L_dz = torch.ones(nclasses, 1)
    local grad_L_W = torch.zeros(nfeatures+1,nclasses)
    
    local c_s = 1
    local it_max = math.floor(Xtrain:size(1)/batch_size)
    
    for ep = 1,ep_max do
        timer = torch.Timer()
        shuffle:copy(torch.randperm(Xtrain:size(1)):type('torch.LongTensor'))
        tot_Loss = 0
        
        for it = 1,it_max do
            -- Initializing gradient for the current iteration
            grad_W:zero()
            grad_b:zero()

            for i = (batch_size*(it-1)+1),batch_size*it do
                --current sentence:
                c_s = shuffle[i]
                --evaluate z:
                z:copy(W:index(1,Xtrain[c_s]:type('torch.LongTensor')):sum(1):add(b)):view(-1,1)
                --evaluate y_hat:
                yhat:copy(torch.exp(z))
                yhat:div(yhat:sum())

                 --evaluate the loss (only on the last iteration for the logger)
                if it == it_max then
                    tot_Loss = tot_Loss - z[Ytrain[c_s]][1] + math.log((z-(z:max())):exp():sum())+z:max()
                end
                --evaluate the gradients:
                -- First with respect to dz (which is equal to db):
                grad_L_dz:copy(yhat)
                grad_L_dz[Ytrain[c_s]]:csub(1)
                --Then with respect to dW:
                grad_L_W:zero()
                grad_L_W:indexAdd(1,Xtrain[c_s]:type('torch.LongTensor'),torch.expand(grad_L_dz,nclasses,53):t())
                grad_L_W[1]:zero()
                --Update the gradients
                grad_W:add(grad_L_W*(1/batch_size))
                grad_b:add(grad_L_dz*1/batch_size)
            end

            -- Apply the regularization every 10 iteratin to gain speed (possible because of the dataset sparsity)
            if (it%10) == 0 then
                W:mul(1-eta*lambda/(W:nElement()+b:nElement())):csub(grad_W:mul(eta))
                b:mul(1-eta*lambda/(W:nElement()+b:nElement())):csub(grad_b:mul(eta))
            else
                W:csub(grad_W:mul(eta))
                b:csub(grad_b:mul(eta))
            end
            -- Updating the loss with the regularization
            if it == it_max then
                tot_Loss = Xtrain:size(1)*tot_Loss/batch_size + (0.5)*lambda*(torch.pow(W,2):sum()+torch.pow(b,2):sum())
            end
        end      
        print('Time elapsed for epoch ' .. ep ..': ' .. timer:time().real .. ' seconds')
        print('Approximative Loss for the last batch for epoch ' .. ep ..': ' .. tot_Loss)
    end
    return W, b, tot_Loss
    
end

--------------------------
-- Linear SVM training
--------------------------
function linearSVM(Xtrain, Ytrain, nfeatures, nclasses, ep_max, batch_size, eta, lambda)
    
    -- Initialization
    local shuffle = torch.LongTensor(Xtrain:size(1))
    local W = torch.cat(torch.zeros(1,nclasses),torch.ones(nfeatures,nclasses),1)
    local b = torch.ones(nclasses, 1)
    local grad_W = torch.zeros(nfeatures+1,nclasses)
    local grad_b = torch.zeros(nclasses, 1)
    local yhat = torch.ones(nclasses, 1)
    local y_temp = torch.zeros(nclasses, 1)
    local grad_L_dy = torch.ones(nclasses, 1)
    local grad_L_W = torch.zeros(nfeatures+1,nclasses)
    
    local c_s = 1
    local it_max = math.floor(Xtrain:size(1)/batch_size)
    
    for ep = 1,ep_max do   
        timer = torch.Timer()
        shuffle:copy(torch.randperm(Xtrain:size(1)):type('torch.LongTensor'))       
        tot_Loss = 0
        
        for it = 1,it_max do

            tot_Loss = 0
            grad_W:zero()
            grad_b:zero()

            for i = (batch_size*(it-1)+1),batch_size*it do
                --current sentence:
                c_s = shuffle[i]
                --evaluate y_hat:
                y_hat = W:index(1,Xtrain[c_s]:type('torch.LongTensor')):sum(1):add(b):view(-1,1)

                --evaluate the loss (only on the last iteration for the logger)
                y_temp:copy(y_hat)
                y_temp[Ytrain[c_s]]:fill(-9999)
                max_c,argmax_c = y_temp:max(1)
                if it == it_max then
                    tot_Loss = tot_Loss + math.max(0,1-y_hat[Ytrain[c_s]][1]+max_c[1][1])
                end
                
                -- Evaluate the gradients
                grad_L_dy:zero()
                if (y_hat[Ytrain[c_s]][1]-max_c[1][1])<1 then
                    grad_L_dy[Ytrain[c_s]] = -1
                    grad_L_dy[argmax_c[1][1]] = 1
                end
                grad_L_W:zero()
                grad_L_W:indexAdd(1,Xtrain[c_s]:type('torch.LongTensor'),torch.expand(grad_L_dy,nclasses,53):t())
                grad_L_W[1]:zero()

                --Update the gradients
                grad_W:add(grad_L_W*(1/batch_size))
                grad_b:add(grad_L_dy*(1/batch_size))

            end
 
            -- Apply the regularization every 10 iteratin to gain speed (possible because of the dataset sparsity)
            if (it%10) == 0 then
                W:mul(1-eta*lambda/(W:nElement()+b:nElement())):csub(grad_W:mul(eta))
                b:mul(1-eta*lambda/(W:nElement()+b:nElement())):csub(grad_b:mul(eta))
            else
                W:csub(grad_W:mul(eta))
                b:csub(grad_b:mul(eta))
            end
            -- Updating the loss with the regularization
            if it == it_max then
                tot_Loss = Xtrain:size(1)*tot_Loss/batch_size + (0.5)*lambda*(torch.pow(W,2):sum()+torch.pow(b,2):sum())
            end
        end
        print('Time elapsed for epoch ' .. ep ..': ' .. timer:time().real .. ' seconds')
        print('Approximative Loss for epoch ' .. ep ..': ' .. tot_Loss) 
    end
    
    return W,b,tot_Loss
end

function nfold_cv(Classifier,Xtrain,Ytrain,nfold,hyperparam,ep_max,nfeatures,nclasses)
    local hyperparam = hyperparam
    local tensor_folds = torch.split(torch.linspace(1,Xtrain:size()[1],Xtrain:size()[1]),nfold)
    local Xtrain_fold = torch.Tensor(math.floor(Xtrain:size()[1])/nfold),Xtrain:size()[2])
    local Xtrain_rest = torch.zeros(math.floor(Xtrain:size()[1])/nfold)*(nfold-1),Xtrain:size()[2])
    local Ytrain_fold = torch.Tensor(math.floor(Xtrain:size()[1])/nfold))
    local Ytrain_rest = torch.zeros(math.floor(Xtrain:size()[1])/nfold)*(nfold-1))
    local av_acc = 0
    
    for n_mod = 1,hyperparam:size()[1] do

        for i = 1,nfold do
            Xtrain_fold:copy(Xtrain:index(1,tensor_folds[i]:type('torch.LongTensor')))
            Ytrain_fold:copy(Ytrain:index(1,tensor_folds[i]:type('torch.LongTensor')))

            for j = 1,nfold do
                if j ~= i then
                    Xtrain_rest:indexAdd(1,torch.linspace((j-1)*nfold+1,j*nfold,nfold):type('torch.LongTensor'),Xtrain:index(1,tensor_folds[j]:type('torch.LongTensor'))
                    Ytrain_rest:indexAdd(1,torch.linspace((j-1)*nfold+1,j*nfold,nfold):type('torch.LongTensor'),Ytrain:index(1,tensor_folds[j]:type('torch.LongTensor'))
                end
            end

            if Classifier == 'NB' then

                prior, x_conditional_y = NaiveBayes(Xtrain_rest, Ytrain_rest, nfeatures, nclasses, hyperparam[n_mod][1])

                Ypred, acc = predict_NB(Xtrain_fold,prior,x_conditional_y,nclasses,Ytrain_fold)


            else

                if Classifier = 'log_reg' then
                    W, b, L = logreg(Xtrain_rest, Ytrain_rest, nfeatures, nclasses, ep_max, hyperparam[n_mod][1], hyperparam[n_mod][2], hyperparam[n_mod][3])
                elseif Classifier = 'linear_svm' then
                    W, b, L = linearSVM(Xtrain_rest, Ytrain_rest, nfeatures, nclasses, ep_max, hyperparam[n_mod][1], hyperparam[n_mod][2], hyperparam[n_mod][3])
                end 

                Validpred, acc = predict(Xtrain_fold, W, b, Ytrain_fold)

            end

            av_acc = av_acc + ac

        end

        if Classifier == 'NB' then
            hyperparam[n_mod][2] = av_acc/nfold
        else
            hyperparam[n_mod][4] = av_acc/nfold

    end

    return hyperparam

end



function main() 
   -- Parse input params
   opt = cmd:parse(arg)
   local f = hdf5.open(opt.datafile, 'r')
   nclasses = f:read('nclasses'):all():long()[1]
   nfeatures = f:read('nfeatures'):all():long()[1]

   local W = torch.DoubleTensor(nclasses, nfeatures)
   local b = torch.DoubleTensor(nclasses)

    -- Train.
    train_input = f:read('train_input'):all()
    train_output = f:read('train_output'):all()

    n = train_output:size(1);

    -- Validation.
    valid_input = f:read('valid_input'):all()
    valid_output = f:read('valid_output'):all()

    -- Test
    test_input = f:read('test_input'):all()

    -- Reading the model
    classifier = opt.classifier
    if (classifier == 'nb') then
        -- Hyperparameters
        alpha = opt.alpha

        -- Learning the model
        prior, x_conditional_y = NaiveBayes(train_input, train_output, nfeatures, nclasses, alpha)
        -- Prediction on the validation set
        Ypred, accuracy = predict_NB(valid_input, prior, x_conditional_y, nclasses, valid_output)
        -- Prediction on the test set
        Testpred = predict_NB(test_input, prior, x_conditional_y, nclasses)
    else 
        -- Hyperparameters
        ep_max = opt.ep_max
        batch_size = opt.batch_size
        eta = opt.eta
        lambda = opt.lambda

        -- Learning the model
        if (classifier == 'log_reg') then
            W, b, L = logreg(train_input, train_output, nfeatures, nclasses, ep_max, batch_size, eta, lambda)
        elseif (classifier == 'linear_svm') then
            W, b, L = linearSVM(train_input, train_output, nfeatures, nclasses, ep_max, batch_size, eta, lambda)
        end
        -- Prediction on the validation set
        Validpred, accuracy = predict(valid_input, W, b, valid_output)
        -- Prediction on the test set
        Testpred = predict(test_input, W, b)
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
