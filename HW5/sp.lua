function train_model(train_input, sent, train_output, observations_dev, model, din, nclass, eta, nEpochs)
    -- Train the model with the structured perceptron approach
    -- V1: only treating the eged leaving the error

    -- For the verbose print
    observations = observations_dev:narrow(2,1,1):narrow(1,1,1000):clone()
    true_classes = observations_dev:narrow(2,16,1):narrow(1,1,1000):squeeze()
    
    -- Memory allocation
    inputs_batch = torch.DoubleTensor(100, din)
    gold_sequence = torch.DoubleTensor(100)
    high_score_seq = torch.DoubleTensor(100)
    grad_pos = torch.zeros(9)
    grad_neg = torch.zeros(9)
    pr1 = torch.zeros(9)
    pr2 = torch.zeros(9)
    
    for i = 1, nEpochs do
        -- timing the epoch
        timer = torch.Timer()
        
        -- mini batch loop
        for t = 2, sent:size(1)-1 do
            -- Mini batch data
            sent_size = sent[{t,2}]
--             print('here1')
            
            inputs_batch:narrow(1,1,sent_size+1):copy(train_input:narrow(1,sent[{t,1}]-1,sent_size+1))
--             print('here2')
            
            gold_sequence:narrow(1,1,sent_size+1):copy(train_output:narrow(1,sent[{t,1}]-1,sent_size+1))
--             print('here3')
            
            -- reset gradients
            model:zeroGradParameters()
            --gradParameters:zero()

            -- Forward pass on a batch subsequence:
            high_score_seq:narrow(1,1,sent_size+1):copy(viterbi(inputs_batch:narrow(1,1,sent_size+1):narrow(2,1,1), 
                                                                compute_logscore, model, nclass))
--             print('here4')
            
            
            for ii = 1, sent_size+1 do
                grad_pos:zero()
                if high_score_seq[ii] ~= gold_sequence[ii] then
                    -- WARNING: Need to call backward right after the forward with the same input to compute correct gradients
                    
                    -- Use of a single gradient (grad_pos) with a penalization on the wrong class predicted (1)
                    -- and a valorisation (-1) on the correct class to predict
                    -- We treat here only the transition after the error
                    model:forward({inputs_batch:narrow(1,ii,1):narrow(2,1,1),inputs_batch:narrow(1,ii,1):narrow(2,2,9)})
                    grad_pos[gold_sequence[ii]] = -1
                    grad_pos[high_score_seq[ii]] = 1
                    model:backward({inputs_batch:narrow(1,ii,1):narrow(2,1,1),inputs_batch:narrow(1,ii,1):narrow(2,2,9)}, grad_pos:view(1,9))
                    
                    
                end
            end
--             print('here7')
            model:updateParameters(eta)
            
        end
            
        print('Epoch '..i..': '..timer:time().real)
        -- Print the f-score on a the first 1000 words to follow the improvement of the model
        cl = viterbi(observations, compute_logscore, model, 9)
        print (f_score(cl, true_classes))
       
    end
end

function train_model2(train_input, sent, train_output, model, din, nclass, eta, nEpochs, obs_val, true_val, f_score)
    -- Train the model with the structured perceptron approach
    -- V2: treating the two edges, the one leading to the error and the
    -- one leaving the error.
    
    val_res = torch.zeros(nEpochs,3)
    -- Memory allocation
    inputs_batch = torch.DoubleTensor(100, din)
    gold_sequence = torch.DoubleTensor(100)
    high_score_seq = torch.DoubleTensor(100)
    grad_pos = torch.zeros(9)
    grad_neg = torch.zeros(9)
    one_hot_true = torch.zeros(1,9)
    one_hot_false = torch.zeros(1,9)
    
    for i = 1, nEpochs do
        -- timing the epoch
        timer = torch.Timer()
        
        -- mini batch loop
        for t = 2, sent:size(1)-1 do
            -- Mini batch data
            sent_size = sent[{t,2}]
--             print('here1')
            
            inputs_batch:narrow(1,1,sent_size+1):copy(train_input:narrow(1,sent[{t,1}]-1,sent_size+1))
--             print('here2')
            
            gold_sequence:narrow(1,1,sent_size+1):copy(train_output:narrow(1,sent[{t,1}]-1,sent_size+1))
--             print('here3')
            
            -- reset gradients
            model:zeroGradParameters()
            --gradParameters:zero()

            -- Forward pass on a batch subsequence:
            high_score_seq:narrow(1,1,sent_size+1):copy(viterbi(inputs_batch:narrow(1,1,sent_size+1):narrow(2,1,1), 
                                                                compute_logscore, model, nclass))
--             print('here4')
            
            previous_error = false

            for ii = 1, sent_size+1 do
                
                grad_neg:zero()
                grad_pos:zero()
                
                if high_score_seq[ii] ~= gold_sequence[ii] and not previous_error then
                    -- WARNING: Need to call backward right after the forward with the same input to compute correct gradients
                    
                    -- Use of a single gradient (grad_pos) with a penalization on the wrong class predicted (1)
                    -- and a valorisation (-1) on the correct class to predict
                    
                    model:forward({inputs_batch:narrow(1,ii,1):narrow(2,1,1),inputs_batch:narrow(1,ii,1):narrow(2,2,9)})
                    grad_pos[gold_sequence[ii]] = -1
                    grad_pos[high_score_seq[ii]] = 1
                    model:backward({inputs_batch:narrow(1,ii,1):narrow(2,1,1),inputs_batch:narrow(1,ii,1):narrow(2,2,9)}, grad_pos:view(1,9))
                    
                    grad_neg:zero()
                    grad_pos:zero()
                    if ii ~= (sent_size + 1) then
                        one_hot_true:zero()
                        one_hot_true[1][gold_sequence[ii]] = 1
                        model:forward({inputs_batch:narrow(1,ii+1,1):narrow(2,1,1),one_hot_true})
                        grad_neg[gold_sequence[ii+1]] = -1
                        model:backward({inputs_batch:narrow(1,ii+1,1):narrow(2,1,1),one_hot_true}, grad_neg:view(1,9) )
                        
                        one_hot_false:zero()
                        one_hot_false[1][high_score_seq[ii]] = 1
                        model:forward({inputs_batch:narrow(1,ii+1,1):narrow(2,1,1),one_hot_false})
                        grad_pos[gold_sequence[ii+1]] = 1
                        model:backward({inputs_batch:narrow(1,ii+1,1):narrow(2,1,1),one_hot_false}, grad_pos:view(1,9) )
                    end
                    
                    previous_error = true
                    
                elseif high_score_seq[ii] ~= gold_sequence[ii] and previous_error then
                    
                    if ii ~= sent_size + 1 then
                        one_hot_true:zero()
                        one_hot_true[1][gold_sequence[ii]] = 1
                        model:forward({inputs_batch:narrow(1,ii+1,1):narrow(2,1,1),one_hot_true})
                        grad_neg[gold_sequence[ii+1]] = -1
                        model:backward({inputs_batch:narrow(1,ii+1,1):narrow(2,1,1),one_hot_true}, grad_neg:view(1,9) )
                        
                        one_hot_false:zero()
                        one_hot_false[1][high_score_seq[ii]] = 1
                        model:forward({inputs_batch:narrow(1,ii+1,1):narrow(2,1,1),one_hot_false})
                        grad_pos[gold_sequence[ii+1]] = 1
                        model:backward({inputs_batch:narrow(1,ii+1,1):narrow(2,1,1),one_hot_false}, grad_pos:view(1,9) )
                    end
                    
                    previous_error = true
                    
                else
                    previous_error = false
                end
            end
--             print('here7')
            model:updateParameters(eta)
            
        end
            
        print('Epoch '..i..': '..timer:time().real)
        cl = viterbi(obs_val, compute_logscore, model, 9)
        val_res[i][1], val_res[i][2], val_res[i][3]  = f_score(cl, true_val)
        print('f-score: '.. val_res[i][1])
        
    end
    return val_res
end