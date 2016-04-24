require 'hdf5';
require 'nn';
require 'torch';
require 'xlua';
require 'randomkit'


-- README:
-- Function to define the 1-hop memory model
-- Inputs: hidden dimension of the lookup table and number of potential answer to predict on
-- Structure is kind of tricky, when calling forward inputs needs to be in the following format:
-- {{{question, story}, story}, question}
-- This is because without nngraph, we need to use mutliple paralleltable at different step.

function buildmodel(hid, nans)

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

function accuracy(sentences, questions, questions_sentences, answers, model)
	local acc = 0
	for i = 1, questions:size(1) do
		xlua.progress(i, questions:size(1))
		story = sentences:narrow(1,questions_sentences[i][1], questions_sentences[i][2]-questions_sentences[i][1]+1)
        input = {{{questions[i], story}, story}, questions[i]}
		pred = model:forward(input)
		m, a = pred:view(7,1):max(1)
		if a[1][1] == answers[i][1] then
			acc = acc + 1
		end
	end
	return acc/questions:size(1)
end


function train_model(sentences, questions, questions_sentences, answers, model, criterion, eta, nEpochs)
    -- Train the model with a SGD
    -- standard parameters are
    -- nEpochs = 1
    -- eta = 0.01

    -- To store the loss
    av_L = 0

    for i = 1, nEpochs do
        -- timing the epoch
        timer = torch.Timer()
        av_L = 0
        if i % 25 == 0 and i < 100 then
        	eta = eta/2
        end
        -- mini batch loop
        for t = 1, questions:size(1) do
        	-- Display progess
            xlua.progress(t, questions:size(1))
            -- define input:
            story = sentences:narrow(1,questions_sentences[t][1], questions_sentences[t][2]-questions_sentences[t][1]+1)
            input = {{{questions[t], story}, story}, questions[t]}

            -- reset gradients
            model:zeroGradParameters()
            --gradParameters:zero()

            -- Forward pass (selection of inputs_batch in case the batch is not full, ie last batch)
            pred = model:forward(input)
            -- Average loss computation
            f = criterion:forward(pred, answers[t])
            av_L = av_L +f

            -- Backward pass
            df_do = criterion:backward(pred, answers[t])
            model:backward(input, df_do)
            model:updateParameters(eta)
            
        end
        acc = accuracy(sentences, questions, questions_sentences, answers, model)
        print('Epoch '..i..': '..timer:time().real)
       	print('\n')
        print('Average Loss: '..av_L/questions:size(1))
        print('\n')
        print('Training accuracy: '..acc)
        print('\n')
        print('***************************************************')
       
    end

end

-- Sanity check:

myFile = hdf5.open('../Data/preprocess/task2_train.hdf5','r')
f = myFile:all()
sentences = f['sentences'] + 1
questions = f['questions'] + 1
questions_sentences = f['questions_sentences']
answers = f['answers']+1
myFile:close()

model = buildmodel(50,7)
parameters, gradParameters = model:getParameters()
randomkit.normal(parameters, 0, 0.1)
criterion = nn.ClassNLLCriterion()

-- i = 1
-- story = sentences:narrow(1,questions_sentences[i][1], questions_sentences[i][2]-questions_sentences[i][1]+1)
-- input = {{{questions[i], story}, story}, questions[i]}
-- pred = model:forward(input)
-- m, a = pred:view(7,1):max(1)
-- print(a)

train_model(sentences, questions, questions_sentences, answers, model, criterion, 0.01, 100)

