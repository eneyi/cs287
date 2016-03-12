----------------
--- helper
----------------

-- Loading train of the gram_size N
function get_train(N)
    local filename = N .. '-grams.hdf5'
    --print(filename)
    myFile = hdf5.open(filename,'r')
    train = myFile:all()['train']
    myFile:close()
    return train
end

function perplexity(distribution, true_words)
    -- exp of the average of the cross entropy of the true word for each line
    -- true words (N_words to predict, one hot true value among 50)
    local perp = 0
    local N = true_words:size(1)
    for i = 1,N do
        mm,aa = true_words[i]:max(1)
        perp = perp + math.log(distribution[{i, aa[1]}])
    end
    perp = math.exp(- perp/N)
    return perp
end


function build_context_count(count_tensor)
    local indexes
    local indexN
    -- Ngram count (depend on w and context)
    -- {'index1-...-indexN-1': {'indexN' : count}}
    local F_c_w = {}
    -- F_c dict (independent of w, only context based)
    -- {index1-...-indexN-1 : count all words in c}
    local F_c = {}
    -- N_c dict (independent of w, only context based)
    -- {index1-...-indexN-1 : count unique type of words in c}
    local N_c = {}

    local N = count_tensor:size(1)
    local M = count_tensor:size(2)

    for i=1, N do
        indexN = count_tensor[{i,M-1}]
        
        -- build the key index1-...-indexN-1
        indexes = tostring(count_tensor[{i,1}])
        for j=2, M - 2 do
            indexes = indexes .. '-' .. tostring(count_tensor[{i,j}])
        end
        
        -- Filling F_c_w
        if F_c_w[indexes] == nil then
            F_c_w[indexes] = {[indexN] = count_tensor[{i, M}]}
        else
            F_c_w[indexes][indexN] = count_tensor[{i, M}]
        end
        
        -- Updating F_c and F_c
        if F_c[indexes] == nil then
            F_c[indexes] = count_tensor[{i, M}]
            N_c[indexes] = 1
        else
            F_c[indexes] = count_tensor[{i, M}] + F_c[indexes]
            N_c[indexes] = 1 + N_c[indexes]
        end
    end
    
    return F_c_w, F_c, N_c
end

----------------
--- Maximum Likekihood Estimation
----------------

function compute_mle_line(N, entry, F_c_w, alpha)
    -- Compute the maximum likelihood estimation with alpha smoothing on the 
    -- input in entry, 
    --
    -- Return vector (50) predicting the distribution from entry
    -- N represent the Ngram size used in the prediction so context is N-1 gram
    local prediction = torch.zeros(50)
    local indexN
    
    -- context (at least with one element)
    local indexes = tostring(entry[{1, entry:size(2)}])
    for j=entry:size(2) - 1, entry:size(2) - 1 - (N-3), -1 do
        indexes = tostring(entry[{1, j}]) .. '-' .. indexes
    end
    -- check if context is unseen, otherwise go to next context
    if F_c_w[indexes] == nil then
        --print('unseen context')
        prediction:fill(alpha)
    else
        -- Compute MLE for each word
        for j=1,50 do
            indexN = entry[{1, j}]
            if F_c_w[indexes][indexN] ~= nil then
                prediction[j] = F_c_w[indexes][indexN] + alpha
            else
                --print('unseen word')
                prediction[j] = alpha
            end
        end
    end

    return prediction:div(prediction:sum())
end

-- Prediction with the MLE (with Laplace smoothing, no back-off and interpolation)

function mle_proba(N, data, alpha)
    -- Output format: distribution predicted for each N word along the
    -- 50 possibilities
    local N_data = data:size(1)
    
    -- Train model
    local train = get_train(N)
    local F_c_w = build_context_count(train)

    -- Prediction
    local distribution = torch.zeros(N_data, 50)
    for i=1, N_data do
        distribution:narrow(1, i, 1):copy(compute_mle_line(N, data:narrow(1,i,1), F_c_w, alpha))
    end
    
    return distribution
end

----------------
--- Witten-Bell
----------------

function compute_wb_line(N, entry, F_c_w_table, alpha)
    -- Compute the interpolated Witten-Bell model where we jump tp lower
    -- order models if the context count is 0 or all the words counts in that
    -- context is 0 also.
    --
    -- Return vector (50) predicting the distribution from entry
    -- N represent the Ngram size used in the prediction so context is N-1 gram
    -- alpha is only used for the MLE without any context
    --
    -- NB: the normalization is done based on the words contained in the first 50
    -- columns of the entry as we are building a distribution on a sub sample of a
    -- dictionnary (so we are using the count only of these words to normalize).
    -- Hence the variable denom and N_c_local
    local prediction = torch.zeros(50)
    local indexN
    local indexes
    local denom
    local N_c_local 
    
    -- case where computation only on the prior
    if N == 1 then
        for j=1,50 do
            indexN = entry[{1, j}]
            -- Corner case when prediction on words not on the dict (case for <s>)
            if F_c_w_table[1][tostring(indexN)] == nil then
                prediction[j] = 0
            else
                prediction[j] = F_c_w_table[1][tostring(indexN)][indexN] + alpha
            end
        end
        -- Normalizing
        return prediction:div(prediction:sum(1)[1])
    else
        -- Compute the MLE for current N
        -- context (at least with one element)
        indexes = tostring(entry[{1, entry:size(2)}])
        for j=entry:size(2) - 1, entry:size(2) - 1 - (N-3), -1 do
            indexes = tostring(entry[{1, j}]) .. '-' .. indexes
        end
        
        -- check if context is unseen, otherwise go to next context
        if F_c_w_table[N][indexes] == nil then
            --print('unseen context')
            return compute_wb_line(N-1, entry, F_c_w_table, alpha)
        end
        
        -- local variable initialization
        denom = 0
        N_c_local = 0
        -- Compute MLE for each word
        for j=1,50 do
            indexN = entry[{1, j}]
            if F_c_w_table[N][indexes][indexN] ~= nil then
                prediction[j] = F_c_w_table[N][indexes][indexN]
                denom = denom + F_c_w_table[N][indexes][indexN] + 1
                N_c_local = N_c_local + 1
            end
        end
        
        -- Check that MLE predicted at least one words, otherwise go to next context
        if prediction:sum(1)[1] == 0 then
            --print('unseen words')
            return compute_wb_line(N-1, entry, F_c_w_table, alpha)
        end
        
        -- Combining with next context
        prediction:add(compute_wb_line(N-1, entry, F_c_w_table, alpha):mul(N_c_local)):div(denom)
        return prediction
    end
end

-- Witten Bell: new version, computation done at once line by line
--
-- p_wb(w|c) = (F_c_w + N_c_. * p_wb(w|c'))/(N_c_. + F_c_.)
function distribution_proba_WB(N, data, alpha)
    local N_data = data:size(1)
    local M = data:size(2)

    -- Building the count matrix for each ngram size lower than N.
    local F_c_w_table = {}
    for i=1,N do
        train = get_train(i)
        F_c_w_table[i] = build_context_count(train)
    end

    -- Vector initialisation
    local distribution = torch.zeros(N_data, 50)
    for i=1,N_data do
        -- Compute witten bell for the whole line i
        distribution:narrow(1, i, 1):copy(compute_wb_line(N, data:narrow(1,i,1), F_c_w_table, alpha))
    end
    return distribution
end


----------------
--- Modified Kneser Ney
----------------

-- Version tailored for modified Kneser-Ney:
-- Modif: now we enable a local computation of D
-- (that will be based on the sub vocabulary used in the validation and tesst)

function build_context_count_split(count_tensor, K)
    -- count_tensor in format (N_words, N + 1):
    -- col1, ..., colN = indexes for the Ngram, colN+1 = N_gram count
    -- K: number of count separate cases (need K > 1, usually K = 3)
    --
    -- Ngram count (depend on w and context)
    -- {'index1-...-indexN-1': {'indexN' : count}}
    local F_c_w = {}
    -- n_table: stores the total number of N_grams ending with indexN
    -- with exact number of occurences stored in their key k:
    -- {k : {'indexN': # N_grams ending with indexN with exactly k occurences}}
    local n_table = {}
    for j=1,K+1 do
        n_table[j] = {}
    end

    local N = count_tensor:size(1)
    local M = count_tensor:size(2)

    for i=1, N do
        local indexN = count_tensor[{i,M-1}]
        
        -- build the key index1-...-indexN-1
        indexes = tostring(count_tensor[{i,1}])
        for j=2, M - 2 do
            indexes = indexes .. '-' .. tostring(count_tensor[{i,j}])
        end
        
        -- Filling F_c_w
        if F_c_w[indexes] == nil then
            F_c_w[indexes] = {[indexN] = count_tensor[{i, M}]}
        else
            F_c_w[indexes][indexN] = count_tensor[{i, M}]
        end
        
        -- Building the key to update the corresponding part of n_table
        if count_tensor[{i, M}] > K then
            key_N_c = K
        else
            key_N_c = count_tensor[{i, M}]
        end
        
        -- Updating n_table
        if count_tensor[{i, M}] <= K + 1 then
            if n_table[count_tensor[{i, M}]][indexN] == nil then
                n_table[count_tensor[{i, M}]][indexN] = 1
            else
                n_table[count_tensor[{i, M}]][indexN] = n_table[count_tensor[{i, M}]][indexN] + 1
            end
        end
    end

    return F_c_w, n_table
end

-- V2: with local normalization on the validation sub vocabulary

function compute_mkn_line(N, entry, F_c_w_table, n_table, alpha, K, D)
    -- Compute the Modified Kneser Ney model where we jump to lower
    -- order models if the context count is 0 or all the words counts in that
    -- context is 0 also.
    --
    -- Return vector (50) predicting the distribution from entry
    -- N represent the Ngram size used in the prediction so context is N-1 gram
    -- alpha is only used for the MLE without any context
    local prediction = torch.zeros(50)
    local indexN
    local F_local
    local N_c_local = {}
    for k=1,K do
        N_c_local[k] = 0
    end
    local n_table_local = {}
    for k=1,K+1 do
        n_table_local[k] = 0
    end
    
    -- case where computation only on the prior
    if N == 1 then
        for j=1,50 do
            indexN = entry[{1, j}]
            -- Corner case when prediction on words not on the dict (case for <s>)
            if F_c_w_table[1][tostring(indexN)] == nil then
                prediction[j] = 0
            else
                prediction[j] = F_c_w_table[1][tostring(indexN)][indexN] + alpha
            end
        end
        -- Normalizing
        return prediction:div(prediction:sum(1)[1])
    else
        -- Compute the MLE for current N
        -- context (at least with one element)
        local indexes = tostring(entry[{1, entry:size(2)}])
        for j=entry:size(2) - 1, entry:size(2) - 1 - (N-3), -1 do
            indexes = tostring(entry[{1, j}]) .. '-' .. indexes
        end
        -- check if context is unseen, otherwise go to next context
        if F_c_w_table[N][indexes] == nil then
            --print('unseen context')
            return compute_mkn_line(N-1, entry, F_c_w_table, n_table, alpha, K, D)
        end

        -- Building local n_table
        for j=1,50 do
            indexN = entry[{1, j}]
            -- Updating local n_table
            for k=1,K+1 do
                -- Possible Case where there is no Ngrams ending with indexN with count of K 
                if n_table[N][k][indexN] ~= nil then
                    n_table_local[k] = n_table_local[k] + n_table[N][k][indexN]
                end
            end
        end

        -- Check no 0 in n_table_local
        for k=1,K+1 do
            if n_table_local[k] == 0 then
                print('0 count in n_table_local for ', indexN, k, N)
                n_table_local[k] = 1
            end
        end
        
        -- Building D (needed to compute prediction rows)
        -- Computing local D

        if D == nil then 
            local Y = n_table_local[1]/(n_table_local[1] + 2*n_table_local[2])
            D = {}
            for k=1,K do
               D[k] = k - (1 + k)*Y*n_table_local[1 + k]/n_table_local[k]
            end
        end

        F_local = 0
        -- Compute curent order level with modified absolute discouting for each word
        for j=1,50 do
            indexN = entry[{1, j}]
            -- case word seen
            if F_c_w_table[N][indexes][indexN] ~= nil then
                -- Building the key for the different case of absolute discounting
                if F_c_w_table[N][indexes][indexN] > K then
                    key_N_c = K
                else
                    key_N_c = F_c_w_table[N][indexes][indexN]
                end
                prediction[j] = F_c_w_table[N][indexes][indexN] - D[key_N_c]
                F_local = F_local + F_c_w_table[N][indexes][indexN]
                N_c_local[key_N_c] = N_c_local[key_N_c] + 1
            end
        end

        -- Check that MLE predicted at least one words, otherwise go to next context
        if prediction:sum(1)[1] == 0 then
            --print('unseen words')
            return compute_mkn_line(N-1, entry, F_c_w_table, n_table, alpha, K, D)
        end
        
        -- Computing factor of lower order model (no denominator because we normalize afterwards)
        local gamma = 0
        for k=1,K do
            if N_c_local[k] ~= nil then
                gamma = gamma + D[k]*N_c_local[k]
            end
        end
        if gamma < 0 then
            --print('gamma error')
            return compute_mkn_line(N-1, entry, F_c_w_table, n_table, alpha, K, D)
        end
        -- Combining with next context
        prediction:add(compute_mkn_line(N-1, entry, F_c_w_table, n_table, alpha, K, D):mul(gamma)):div(F_local)
        -- Normalization
        -- TODO: why??? We normalize at the end
        -- prediction:div(prediction:sum(1)[1])
        return prediction
    end
end

-- Modified Kneser Ney: computation done at once line by line
--
-- p_wb(w|c) = (F_c_w + N_c_. * p_wb(w|c'))/(N_c_. + F_c_.)
function distribution_proba_mKN(N, data, alpha, K, D)
    local N_data = data:size(1)
    local M = data:size(2)

    -- Building the count matrix for each ngram size lower than N.
    local F_c_w_table = {}
    local n_table = {}
    for i=1,N do
        train = get_train(i)
        F_c_w_table[i], n_table[i] = build_context_count_split2(train, K)
    end

    -- Vector initialisation
    local distribution = torch.zeros(N_data, 50)
    for i=1,N_data do
        -- Compute witten bell for the whole line i
        distribution:narrow(1, i, 1):copy(compute_mkn_line2(N, data:narrow(1,i,1), F_c_w_table, n_table, alpha, K, D))
    end
    --distribution:cdiv(distribution:sum(2):expand(distribution:size(1), distribution:size(2)))
    return distribution
end