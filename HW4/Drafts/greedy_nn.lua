require 'nn';
require 'hdf5';
require 'neuralnetpart1.lua';

function predict_space(nnlm, dat):

	local dat_nospace = dat
	local it = 1
	local i = 1
	local nextpred = torch.Tensor(2)

	while it<data['input_data_dat_nospace_nospace']:size(1)-(Nwin-1) do
		it = it + 1
		nextpred:copy(nnlm:forward(dat_nospace:narrow(1,i,Nwin)));
		m, argm = nextpred:max(1)

		if argm[1] == 2 then
			i = i + 1		
		elseif argm[1] == 1 then 
			local dat_nospace_ = torch.LongTensor(dat_nospace:size(1)+1)
			dat_nospace_:narrow(1,1,i+(Nwin-1)):copy(dat_nospace:narrow(1,1,i+(Nwin-1)))
			dat_nospace_[i+Nwin] = 1
			dat_nospace_:narrow(1,i+(Nwin-1)+2,dat_nospace:size(1)-i-(Nwin-1)):copy(dat_nospace:narrow(1,i+(Nwin-1)+1,dat_nospace:size(1)-i-(Nwin-1)))
			dat_nospace = dat_nospace_
			i = i + 2
		end 
	end
	return dat_nospace
end

function predict_kaggle(dat_space):
	local num_sent = 0

	for i = 5,dat_space:size(1) do
    	if dat_space[i] == 2 then
        	num_sent = num_sent + 1
    	end
	end

	local num_spaces = torch.DoubleTensor(num_sent,2)
	local row = 1
	local count_space = 0

	for i=5,dat_space:size(1) do
	    if dat_space[i] == 2 then
	        num_spaces[{row, 1}] = row
	        num_spaces[{row, 2}] = count_space
	        count_space = 0
	        row = row + 1
	    elseif dat_space[i] == 1 then
	        count_space = count_space + 1
	    end
	end

	return num_spaces
end

myFile = hdf5.open('data_preprocessed/6-grams.hdf5','r')
data = myFile:all()
myFile:close()

Nwin = 5

train_input = data['input_matrix_train']
train_output = data['output_matrix_train']

valid = data['input_data_valid_nospace']:clone()

test = data['input_data_test']:clone()

torch.manualSeed(1)

nnlm1, crit = build_model(Nwin, 49, 2, 20, 80)

train_model(train_input, train_output, nnlm1, crit, Nwin, 2, 0.01, 28, 10)

valid_space = predict_space(nnlm1, valid)
num_spaces_valid = predict_kaggle(valid_space)

test_space = predict_space(nnlm1, test)
num_spaces_test = predict_kaggle(test_space)

myFile = hdf5.open('../submission/pred_test_greedy_nn_5', 'w')
myFile:write('num_spaces', num_spaces_test)
myFile:close()
