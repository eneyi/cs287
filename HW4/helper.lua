function get_kaggle_format(predictions_test, N)
    -- Counting sentences
    local num_sentence = 0
    for i=N-1,predictions_test:size(1) do
        if predictions_test[i] == 2 then
            num_sentence = num_sentence + 1
        end
    end

    -- Counting space per sentence
    local num_spaces = torch.DoubleTensor(num_sentence,2)
    local row = 1
    local count_space = 0
    for i=N-1,predictions_test:size(1) do
        if predictions_test[i] == 2 then
            num_spaces[{row, 1}] = row
            num_spaces[{row, 2}] = count_space
            count_space = 0
            row = row + 1
        elseif predictions_test[i] == 1 then
            count_space = count_space + 1
        end
    end
    return num_spaces
end

function compute_rmse(true_kaggle, pred_kaggle)
    local rmse = 0
    for i=1,true_kaggle:size(1) do
        rmse = rmse + math.pow(true_kaggle[{i,2}] - pred_kaggle[{i,2}], 2)
    end
    return(math.sqrt(rmse/ true_kaggle:size(1)))
end