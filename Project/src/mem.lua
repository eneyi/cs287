require 'hdf5';
require 'nn';
require 'torch';


-- README:
-- Function to define the 1-hop memory model
-- Inputs: hidden dimension of the lookup table and number of potential answer to predict on
-- Structure is kind of tricky, when calling forward inputs needs to be in the following format:
-- {{{question, story}, story}, question}
-- This is because without nngraph, we need to use mutliple paralleltable at different step.

function buildmodel(hid, nans)
	--Lookup dimension:
	hid = 5
	nans = 6

	-- Initialise the 3 lookup tables:
	question_embedding = nn.Sequential();
	question_embedding:add(nn.LookupTable(torch.max(sentences), hid));
	question_embedding:add(nn.Sum(1));
	question_embedding:add(nn.View(1, hid));

	sent_input_embedding = nn.Sequential();
	sent_input_embedding:add(nn.LookupTable(torch.max(sentences), hid));
	sent_input_embedding:add(nn.Sum(2));

	sent_output_embedding = nn.Sequential();
	sent_output_embedding:add(nn.LookupTable(torch.max(sentences), hid));
	sent_output_embedding:add(nn.Sum(2));

	-- Define the inner product + softmax between input and question:
	inner_prod = nn.Sequential();
	PT = nn.ParallelTable();
	PT:add(question_embedding);
	PT:add(sent_input_embedding);
	inner_prod:add(PT);
	inner_prod:add(nn.MM(false, true));
	inner_prod:add(nn.SoftMax());

	-- Define the weighted sum:
	weighted_sum = nn.MM();

	-- Define the part of the model that yields the o vector using a weighted sum:
	model_inner = nn.Sequential();
	model_pt_inner = nn.ParallelTable();
	model_pt_inner:add(inner_prod);
	model_pt_inner:add(sent_output_embedding);

	model_inner:add(model_pt_inner);
	model_inner:add(weighted_sum);

	-- Building the model itself:
	model = nn.Sequential();
	model_pt = nn.ParallelTable();

	-- Adding the part leading to o:
	model_pt:add(model_inner);
	-- Adding the part leading to u:
	model_pt:add(question_embedding);

	model:add(model_pt);

	-- Summing o and u:
	model:add(nn.JoinTable(1));
	model:add(nn.Sum(1));

	-- Applying a linear transformation W without bias term
	model:add(nn.Linear(hid, nans, false));

	-- Applying a softmax function to obtain a distribution over the possible answers
	model:add(nn.LogSoftMax());

	return model
end

-- Sanity check:

myFile = hdf5.open('../Data/preprocess/task2_train.hdf5','r')
f = myFile:all()
sentences = f['sentences'] + 1
questions = f['questions'] + 1
questions_sentences = f['questions_sentences']
answers = f['answers']+1
myFile:close()

model = buildmodel(5,6)
criterion = nn.ClassNLLCriterion()

input = {{{questions[1], sentences:narrow(1,1,2)}, sentences:narrow(1,1,2)}, questions[1]}
preds = model:forward(input)
L = criterion:forward(preds, answers[1])
dL = criterion:backward(preds, answers[1])
model:backward(input, dL)

