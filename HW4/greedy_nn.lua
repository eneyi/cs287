require 'nn';
require 'hdf5';
require 'neuralnetpart1.lua';

myFile = hdf5.open('data_preprocessed/5-grams.hdf5','r')
data = myFile:all()
myFile:close()

Nwin = 4

train_input = data['input_matrix_train']
train_output = data['output_matrix_train']

nnlm1, crit = build_model(Nwin, 49, 2, 20, 80)

train_model(train_input, train_output, nnlm1, crit, Nwin, 2, 0.01, 28, 10)
valid = data['input_data_valid_nospace']:clone()

it = 1
i = 1
nextpred = torch.Tensor(2)

print('here')
while it<data['input_data_valid_nospace']:size(1)-(Nwin-1) do
	if it % 50 == 0 then
		print( it..'/'..data['input_data_valid_nospace']:size(1))
	end
	it = it + 1
	nextpred:copy(nnlm1:forward(valid:narrow(1,i,Nwin)));
	m, argm = nextpred:max(1)

	if argm[1] == 2 then
		i = i + 1		
	elseif argm[1] == 1 then 
		valid_ = torch.LongTensor(valid:size(1)+1)
		valid_:narrow(1,1,i+(Nwin-1)):copy(valid:narrow(1,1,i+(Nwin-1)))
		valid_[i+Nwin] = 1
		valid_:narrow(1,i+(Nwin-1)+2,valid:size(1)-i-(Nwin-1)):copy(valid:narrow(1,i+(Nwin-1)+1,valid:size(1)-i-(Nwin-1)))
		valid = valid_
		i = i + 2
	end 
end

print(valid:narrow(1,1,25))
print(data['input_data_valid']:narrow(1,1,30))