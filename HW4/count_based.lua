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
cmd:option('-N', 2, 'Ngram size for the input')
cmd:option('-algo', 'greedy', 'Algorithm to use: either greedy or viterbi')
cmd:option('-f', 'pred_test.f5', 'File name for the predictions on the test')

-- Build the mapping from (N-1)gram to row index
-- and the count matrix F_count: (num_context, 2)
function get_F_count(F, N)
    local ngram_to_ind = {}
    local key
    for i=1,F:size(1) do
        key = tostring(F[{i,1}])
        -- Building key
        for k = 2,N-1 do
            key = key .. '-' .. tostring(F[{i,k}])
        end
        ngram_to_ind[key] = i
    end
    return F:narrow(2,N,2), ngram_to_ind
end

-- Compute proba distribution over (space, char) for the context
-- F is here the count matrix (num_context, 2)
function compute_count_based_probability(context, F_count, ngram_to_ind, alpha)
    local probability = torch.zeros(2)
    -- Building key, ie (N-1)gram (from i to i+(N-2))
    local key = tostring(context[1])
    for k = 2,context:size(1) do
        key = key .. '-' .. tostring(context[k])
    end
    -- If (N-1)gram never seen, prior distribution
    if (ngram_to_ind[key] ~= nil) then
        -- index of the current (n-1)gram in the F matrix
        local index = ngram_to_ind[key]
        probability:copy(F_count:narrow(1,index,1))
        -- Adding smoothing
        probability:add(alpha)
    -- Case unseen context
    else
        -- Prior
        probability:copy(torch.DoubleTensor({F_count:narrow(2,1,1):sum(), F_count:narrow(2,2,1):sum()}))
    end
    return probability:div(probability:sum())
end

-- Compute perplexity on entry with space
function compute_perplexity(gram_input, F_count, ngram_to_ind, N)
    local perp = 0
    local context = torch.zeros(N-1)
    local probability = torch.zeros(2)
    -- Do not predict for the last char
    --for i=1,gram_input:size(1)-N do
    local size=gram_input:size(1) - (N-1)
    for i=1,size do
        context:copy(gram_input:narrow(1,i,N-1))
        -- Line where the model appears
        probability:copy(compute_count_based_probability(context, F_count, ngram_to_ind, 1))
        if gram_input[i+(N-1)] == 1 then
            right_proba = probability[1]
            --print('space')
            --print(right_proba)
        else
            right_proba = probability[2]
        end
        perp = perp + math.log(right_proba)
    end
    perp = math.exp(-perp/size)
    --perp = math.exp(-perp/(gram_input:size(1)-N))
    return perp
end

-- Greedy algorithm to predict a sequence from gram_input with a count
-- based probability model
function predict_count_based_greedy(gram_input, F_count, ngram_to_ind, N)
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
        probability:copy(compute_count_based_probability(context, F_count, ngram_to_ind, 1))
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

-- Viterbi algorithm to predict a sequence from gram_input with a count
-- based probability model
-- pi matrix format (col1: space; col2: char)
function predict_count_based_viterbi(gram_input, F_count, ngram_to_ind, N)
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
            y_hat:copy(compute_probability(context, F_count, ngram_to_ind, 1))

            for c_current =1,2 do
                score = pi[{i-1, c_prev}] + math.log(y_hat[c_current])
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

-- Viterbi trigram
function predict_count_based_viterbi_trigram(gram_input, F_count, ngram_to_ind, N)
    -- Backpointer
    local score
    local bp = torch.zeros(gram_input:size(1) + 1, 3)
    local context = torch.DoubleTensor(2)
    local y_hat = torch.DoubleTensor(2)
    -- pi is built as ('char-space', 'char-char', 'space-char')
    -- corresponding index in the context
    local pi = torch.ones(gram_input:size(1) + 1, 3):mul(-999999999)
    -- Initialization
    pi[{2,1}] = 0
    --pi[{2,2}] = 0
    --pi[{2,3}] = 0
    -- We need to start at the first trigram
    for i=3,gram_input:size(1)+1 do
        for c_prev =1,3 do
            -- Precompute y_hat(c_prev)
            if c_prev == 1 then
                context[1] = gram_input[i-2]
                context[2] = 1
            elseif c_prev == 2 then
                context[1] = gram_input[i-2]
                context[2] = gram_input[i-1]
            else
                context[1] = 1
                context[2] = gram_input[i-1]
            end
            -- Line where the model appears
            y_hat:copy(compute_probability(context, F_count, ngram_to_ind, 1))

            -- cannot have 2 spaces in a row: from 1 goes to 3 necessarily
            if c_prev == 1 then
                pi[{i, 3}] = pi[{i-1, c_prev}] + math.log(y_hat[2])
                bp[{i, 3}] = c_prev
            else
                -- last char is necessarily 'char' so
                -- 1: space predicted (ie 'char-space')
                -- 2: char predicted (ie 'char-char')
                for c_current =1,2 do
                    score = pi[{i-1, c_prev}] + math.log(y_hat[c_current])
                    if score > pi[{i, c_current}] then
                        pi[{i, c_current}] = score
                        bp[{i, c_current}] = c_prev
                    end
                end
            end
        end
    end
    return pi, bp
end

-- Building the sequences from the backpointer
-- We start the sequence by the ('char'-'char') configuration
-- as we know it's the only one possible
function build_sequences_from_bp_trigram(bp, gram_input)
    local predictions = torch.DoubleTensor(2*gram_input:size(1))
    -- Next position to fill in predictions (have to do it backward)
    local position = 2*gram_input:size(1)
    local col = 2
    -- Loop until the 4th position 
    for i=bp:size(1),4,-1 do
        -- coming from a space
        if bp[i][col] == 1 then
            predictions[position] = 1
            position = position - 1
        end
        col = bp[i][col]
        -- index i is shifted of 1 wrt local index in gram_input
        predictions[position] = gram_input[i-1]
        position = position - 1
    end
    -- Beginnning of gram_input set
    predictions[position] = gram_input[2]
    position = position - 1
    predictions[position] = gram_input[1]
    position = position - 1

    return predictions:narrow(1,position+1,predictions:size(1)-position)
end

function main() 
    -- Parse input params
    opt = cmd:parse(arg)
    N = opt.N
    algo = opt.algo

    -- Reading file
    local file = hdf5.open('data_preprocessed/'..tostring(N)..'-grams.hdf5', 'r')
    data = file:all()
    file:close()

    F_train = data['F_train']
    input_data_valid = data['input_data_valid']
    input_data_train = data['input_data_train']
    input_data_test = data['input_data_test']
    input_data_valid_nospace = data['input_data_valid_nospace']
    
    -- Building the model
    F_count, ngram_to_ind = get_F_count(F_train, N)
    print('Ngram size '..tostring(N))
    print('Train Perplexity')
    print(compute_perplexity(input_data_train, F_count, ngram_to_ind, N))
    print('Valid Perplexity')
    print(compute_perplexity(input_data_valid, F_count, ngram_to_ind, N))

    -- Prediction
    if (algo == 'greedy') then
        predictions_test = predict_count_based_greedy(input_data_test, F_count, ngram_to_ind, N)
    elseif (algo == 'viterbi') then
        if (N == 2) then
            pi, bp = predict_count_based_viterbi(input_data_test, F_count, ngram_to_ind, N)
            predictions_test = build_sequences_from_bp(bp, input_data_test)
        elseif (N == 3) then
            pi_tri, bp_tri = predict_count_based_viterbi_trigram(input_data_test, F_count, ngram_to_ind, N)
            predictions_test = build_sequences_from_bp_trigram(bp_tri, input_data_test)
        else
            error("invalid N for Viterbi")
        end 
    else
        error("invalid algorithm input") 
    end

    -- Kaggle format
    num_spaces = get_kaggle_format(predictions_test, N)

    -- Saving the Kaggle format output
    myFile = hdf5.open('submission/'..opt.f, 'w')
    myFile:write('num_spaces', num_spaces)
    myFile:close()
end

main()
