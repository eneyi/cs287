-- function to evaluate the predicted sequence
-- need to compute precision and recall (class 1 stands for negative class)
function compute_score(predicted_classes, true_classes)
    local n = predicted_classes:size(1)
    local right_pred = 0
    local positive_true = 0
    local positive_pred = 0
    for i=1,n do
        if predicted_classes[i] > 1 then
            positive_pred = positive_pred + 1
        end
        if true_classes[i] > 1 then
            positive_true = positive_true + 1
        end
        if (true_classes[i] == predicted_classes[i]) and true_classes[i] > 1 then
            right_pred = right_pred + 1
        end
    end
    local precision = right_pred/positive_pred
    local recall = right_pred/positive_true
    return precision, recall
end
        
function f_score(predicted_classes, true_classes)
    local p,r = compute_score(predicted_classes, true_classes)
    print('Precision: '..p)
    print ('Recall: '..r)
    print ('f-score: '..2*p*r/(p+r))
    return 2*p*r/(p+r)
end