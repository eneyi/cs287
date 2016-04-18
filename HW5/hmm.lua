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
require 'helper.lua';

cmd = torch.CmdLine()

-- Cmd Args
cmd:option('-datafile', 'data/words_feature.hdf5',
           'Datafile with features in hdf5 format')
cmd:option('-alpha_t', 0.1, 'Smoothing parameter alpha in the transition counts')
cmd:option('-alpha_w', 0.1, 'Smoothing parameter alpha in the word counts')
cmd:option('-alpha_c', 8, 'Smoothing parameter alpha in the caps counts')
cmd:option('-alpha_p', 2, 'Smoothing parameter alpha in the pos counts')
cmd:option('-test', 0, 'Boolean (as int) to ask for a prediction on test, will be saved in submission in hdf5 format')
cmd:option('-datafile_test', 'submission/v_seq_hmm', 'Smoothing parameter alpha in the word counts')
cmd:option('-nfeatures', 2, 'Number of type of features to use')
cmd:option('-cv', 0, 'Boolean (as int) to run a cross validation pipeline')



-- Formating as log-probability and smoothing the input
function format_matrix(matrix, alpha)
    local formatted_matrix = matrix:clone():type('torch.DoubleTensor')
    formatted_matrix:add(alpha)
    -- Normalize
    local norm_mat = torch.expandAs(formatted_matrix:sum(1), formatted_matrix)
    formatted_matrix:cdiv(norm_mat)
    return formatted_matrix:log()
end
    
-- log-scores of transition and emission
-- corresponds to the vector y in the lecture notes
-- i: timestep for the computed score
function score_hmm(observations, i, emissions, transition, C, nfeatures)
    local observation_emission = torch.zeros(C)
    for k=1,nfeatures do
        -- print(i,k)
        -- print(emissions[k][observations[{i,k}]])
        observation_emission:add(emissions[k][observations[{i,k}]])
    end
    observation_emission = observation_emission:view(C, 1):expand(C, C)
    -- NOTE: allocates a new Tensor
    return observation_emission + transition
end

-- Viterbi algorithm.
-- observations: a sequence of observations, represented as integers
-- logscore: the edge scoring function over classes and observations in a history-based model
function viterbi(observations, logscore, emissions, transition, nfeatures)
    local y
    -- Formating tensors
    local initial = torch.zeros(transition:size(2), 1)
    -- initial started with a start of sentence: <t>
    initial[{8,1}] = 1
    initial:log()

    -- number of classes
    C = initial:size(1)
    local n = observations:size(1)
    local max_table = torch.Tensor(n, C)
    local backpointer_table = torch.Tensor(n, C)

    -- first timestep
    -- the initial most likely paths are the initial state distribution
    -- NOTE: another unnecessary Tensor allocation here
    local init_pred = initial:clone()
    for i=1,nfeatures do
        init_pred:add(emissions[i][observations[{1,i}]])
    end
    local maxes, backpointers = init_pred:max(2)
    max_table[1] = maxes

    -- remaining timesteps ("forwarding" the maxes)
    for i=2,n do
        -- precompute edge scores
        y = logscore(observations, i, emissions, transition, C, nfeatures)
        scores = y + maxes:view(1, C):expand(C, C)

        -- compute new maxes (NOTE: another unnecessary Tensor allocation here)
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

-- Prediction pipeline
function predict(observations, emissions, transition, alphas, nfeatures)
    -- Formating model parameters (log and alpha smoothing)
    -- Alphas is a tensor : {alpha_t, alpha_w, alpha_c}
    emissions_cleaned = {}
    for i=1,nfeatures do
        emissions_cleaned[i] = format_matrix(emissions[i], alphas[i+1])
    end
    local transition_cleaned = format_matrix(transition, alphas[1])
    
    return viterbi(observations, score_hmm, emissions_cleaned, transition_cleaned, nfeatures)
end

-- Cross validation pipeline
function cross_validation(observations, emissions, transitions, true_classes,
                          alphas_table, alpha_t)
    -- alphas_table is a table of tensor with the range of parameters to use
    -- Current implementation for 2 features only
    -- alphas_table = {alpha_w_tensor, alpha_c_tensor}
    -- Return a tensor with first columns the alpha value and last the score for each
    local nfeatures = #alphas_table
    local v_seq_dev, precision, recall, f
    local alphas = torch.DoubleTensor(3)
    local size1 = alphas_table[1]:size(1)
    local size2 = alphas_table[2]:size(1)
    local num_evaluations = size1*size2

    -- Columns for 2 features are (alphas_w_value, alphas_c_value, f_score, precision, recall)
    local scores = torch.DoubleTensor(num_evaluations, nfeatures+3)

    for i=1,size1 do
        alpha_w = alphas_table[1][i]
        for k=1,size2 do
            alpha_c = alphas_table[2][k]

            alphas:copy(torch.Tensor({alpha_t, alpha_w, alpha_c}))
            v_seq_dev = predict(observations, emissions, transition, alphas, nfeatures)
            precision, recall = compute_score(v_seq_dev, true_classes)
            f = f_score(precision, recall)

            -- Filling the scores tensor
            scores[{(i-1)*size2+k, 1}] = alpha_w
            scores[{(i-1)*size2+k, 2}] = alpha_c
            scores[{(i-1)*size2+k, 3}] = f
            scores[{(i-1)*size2+k, 4}] = precision
            scores[{(i-1)*size2+k, 5}] = recall
        end
    end

    return scores
end


function main() 
    -- Parse input params
    opt = cmd:parse(arg)

    -- Reading file from pre-processing
    myFile = hdf5.open(opt.datafile,'r')
    data = myFile:all()
    emission_w = data['emission_w']
    emission_c = data['emission_c']
    emission_p = data['emission_p']
    print(emission_p:size())
    -- Table of emission tensor (one tensor per feature)
    emissions = {emission_w, emission_c, emission_p}
    -- Assertion on number of features
    nfeatures = opt.nfeatures
    if nfeatures > #emissions then
        error('Too many features specified')
    end
    print('Number of features used: '..nfeatures)
    transition = data['transition']
    input_matrix_train = data['input_matrix_train']
    input_matrix_dev = data['input_matrix_dev']
    input_matrix_test = data['input_matrix_test']
    myFile:close()

    -- Parameters:
    true_classes = input_matrix_dev:narrow(2,6,1):clone():view(input_matrix_dev:size(1))
    -- contain in each column feature observation
    -- (same order as the feature emission tensor in the emissoins table)
    observations = input_matrix_dev:narrow(2,3,nfeatures):clone()
    -- Alpha parameter
    alphas = torch.Tensor({opt.alpha_t, opt.alpha_w, opt.alpha_c, opt.alpha_p})

    -- Prediction on dev
    v_seq_dev = predict(observations, emissions, transition, alphas, nfeatures)
    print(v_seq_dev:size(1))
    precision, recall = compute_score(v_seq_dev, true_classes)
    f = f_score(precision, recall)

    print('Prediction on dev')
    print('Precision is : '..precision)
    print('Recall is : '..recall)
    print('F score (beta = 1) is : '..f)

    -- Cross validation
    if (opt.cv == 1) then
        alphas_table = {}
        -- alpha_w
        alphas_table[1] = torch.Tensor({0.05, 0.1, 0.2, 0.3, 0.5, 0.8})
        -- alpha_c
        alphas_table[2] = torch.Tensor({5, 8, 10, 12, 15, 20, 22})

        scores = cross_validation(observations, emissions, transitions, true_classes,
                                  alphas_table, opt.alpha_t)
        print(scores)

        -- Saving the score
        myFile = hdf5.open('plot_scores.hdf5', 'w')
        myFile:write('scores', scores)
        myFile:close()
        print('CV on dev saved in '..'plot_scores.hdf5')
    end

    -- Prediction on test
    if (opt.test == 1) then 
        print('Prediction on test')
        observations_test = input_matrix_test:narrow(2,3,nfeatures):clone()
        v_seq_test = predict(observations_test, emissions, transition, alphas, nfeatures)
        -- Saving predicted sequence on test
        myFile = hdf5.open(opt.datafile_test, 'w')
        myFile:write('v_seq_test', v_seq_test)
        myFile:write('v_seq_dev', v_seq_dev)
        myFile:close()
        print('Sequence predicted on test saved in '..opt.datafile_test)
    end

end

main()