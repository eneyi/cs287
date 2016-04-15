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
cmd:option('-alpha_t', 1, 'Smoothing parameter alpha in the transition counts')
cmd:option('-alpha_w', 2, 'Smoothing parameter alpha in the word counts')
cmd:option('-alpha_c', 0.5, 'Smoothing parameter alpha in the caps counts')
cmd:option('-test', 0, 'Boolean (as int) to ask for a prediction on test, will be saved in submission in hdf5 format')
cmd:option('-datafile_test', 'submission/v_seq_hmm', 'Smoothing parameter alpha in the word counts')


-- Formating as log-probability and smoothing the input
function format_matrix(matrix, alpha)
    local formatted_matrix = matrix:clone()
    formatted_matrix:add(alpha)
    -- Normalize
    local norm_mat = torch.Tensor(torch.expandAs(formatted_matrix:sum(1), formatted_matrix))
    formatted_matrix:cdiv(norm_mat)
    return formatted_matrix:log()
end
    
-- log-scores of transition and emission
-- corresponds to the vector y in the lecture notes
-- i: timestep for the computed score
function score_hmm(observations, i, emission_w, emission_c, transition, C)
    local observation_emission = emission_w[observations[i]]
    observation_emission:add(emission_c[observations[i]])
    observation_emission:view(C, 1):expand(C, C)
    -- NOTE: allocates a new Tensor
    return observation_emission + transition
end

-- Viterbi algorithm.
-- observations: a sequence of observations, represented as integers
-- logscore: the edge scoring function over classes and observations in a history-based model
function viterbi(observations, logscore, emission_w, emission_c, transition)
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
    local maxes, backpointers = (initial + emission_w[observations[1]] + emission_c[observations[1]]):max(2)
    max_table[1] = maxes

    -- remaining timesteps ("forwarding" the maxes)
    for i=2,n do
        -- precompute edge scores
        y = logscore(observations, i, emission_w, emission_c, transition, C)
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
function predict(observations, emission_w, emission_c, transition, alphas)
    -- Formating model parameters (log and alpha smoothing)
    -- Alphas is a tensor : {alpha_t, alpha_w, alpha_c}
    print(emission_w.size())
    local transition_cleaned = format_matrix(transition, alphas[1])
    local emission_w_cleaned = format_matrix(emission_w, alphas[2])
    local emission_c_cleaned = format_matrix(emission_c, alphas[3])
    
    return viterbi(observations, score_hmm, emission_w_cleaned,
                   emission_c_cleaned, transition_cleaned)
end

function main() 
    -- Parse input params
    opt = cmd:parse(arg)

    -- Reading file

    myFile = hdf5.open(opt.datafile,'r')
    data = myFile:all()
    emission_w = data['emission_w']
    emission_c = data['emission_c']
    transition = data['transition']
    input_matrix_train = data['input_matrix_train']
    input_matrix_dev = data['input_matrix_dev']
    input_matrix_test = data['input_matrix_test']
    myFile:close()

    -- Prediction on dev
    true_classes = input_matrix_dev:narrow(2,4,1):clone():view(input_matrix_dev:size(1))
    observations = input_matrix_dev:narrow(2,3,1):clone():view(input_matrix_dev:size(1))
    -- Alpha parameter
    alphas = torch.Tensor({opt.alpha_t, opt.alpha_w, opt.alpha_c})

    v_seq_dev = predict(observations, emission_w, emission_c, transition, alphas)
    precision, recall = compute_score(v_seq_dev, true_classes)
    f = f_score(precision, recall)

    print('Prediction on dev')
    print('Precision is : '..precision)
    print('Recall is : '..recall)
    print('F score (beta = 1) is : '..f)

    -- Prediction on test
    if (opt.test == 1) then 
        print('Prediction on test')
        observations_test = input_matrix_test:narrow(2,3,1):clone():view(input_matrix_test:size(1))
        v_seq_test = predict(observations_test, emission_w, emission_c, transition, alphas)
        -- Saving predicted sequence on test
        myFile = hdf5.open(opt.datafile_test, 'w')
        myFile:write('v_seq_test', v_seq_test)
        myFile:write('v_seq_dev', v_seq_dev)
        myFile:close()
        print('Sequence predicted on test saved in '..opt.datafile_test)
    end

end

main()