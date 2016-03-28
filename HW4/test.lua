require 'nn';
require 'hdf5';
require 'neuralnetpart1.lua';

myFile = hdf5.open('data_preprocessed/5-grams.hdf5','r')
data = myFile:all()
myFile:close()


train_input = data['input_matrix_train']
train_output = data['output_matrix_train']

nnlm1, crit = build_model(4, 49, 2, 16, 80)

train_model(train_input, train_output, nnlm1, crit, 4, 2, 0.01, 1, 32)
valid = data['input_data_valid_nospace']:clone()

it = 1
i = 1
nextpred = torch.Tensor(2)

print('here')
while it<data['input_data_valid_nospace']:size(1)-3 do
	print( it..'/'..data['input_data_valid_nospace']:size(1))
	it = it + 1
	nextpred:copy(nnlm1:forward(valid:narrow(1,i,4)));
	m, argm = nextpred:max(1)

	if argm[1] == 2 then
		i = i + 1		
	elseif argm[1] == 1 then 
		valid_ = torch.LongTensor(valid:size(1)+1)
		valid_:narrow(1,1,i+3):copy(valid:narrow(1,1,i+3))
		valid_[i+3+1] = 1
		valid_:narrow(1,i+3+2,valid:size(1)-i-3):copy(valid:narrow(1,i+3+1,valid:size(1)-i-3))
		valid = valid_
		i = i + 2
	end 
end