require 'hdf5';
require 'nn';
require 'torch';
require 'xlua';
require 'randomkit';


-- README:
-- Function to define the 1-hop memory model
-- Inputs: hidden dimension of the lookup table and number of potential answer to predict on
-- Structure is kind of tricky, when calling forward inputs needs to be in the following format:
-- {{{question, {memory, torch.linspace(1,memsize,memsize):type('torch.LongTensor')},
-- 							{memory,torch.linspace(1,memsize,memsize):type('torch.LongTensor')}, question}
-- This is because without nngraph, we need to use mutliple paralleltable at different step.

function buildmodel(hid, nvoc, nans, memsize)

	-- Initialise the 3 lookup tables:
	question_embedding = nn.Sequential();
	question_embedding:add(nn.LookupTable(nvoc, hid));
	question_embedding:add(nn.Sum(1));
	question_embedding:add(nn.View(1, hid));

	sent_input_embedding_time = nn.Sequential();
	sent_input_pt = nn.ParallelTable();
	sent_input_embedding = nn.Sequential();
	sent_input_embedding:add(nn.LookupTable(nvoc, hid));
	sent_input_embedding:add(nn.Sum(2));
	TA = nn.LookupTable(memsize,hid);
	sent_input_pt:add(sent_input_embedding);
	sent_input_pt:add(TA);
	sent_input_embedding_time:add(sent_input_pt);
	sent_input_embedding_time:add(nn.CAddTable());


	sent_output_embedding_time = nn.Sequential();
	sent_output_pt = nn.ParallelTable();
	sent_output_embedding = nn.Sequential();
	sent_output_embedding:add(nn.LookupTable(nvoc, hid));
	sent_output_embedding:add(nn.Sum(2));
	TC = nn.LookupTable(memsize,hid);
	sent_output_pt:add(sent_output_embedding);
	sent_output_pt:add(TC);
	sent_output_embedding_time:add(sent_output_pt);
	sent_output_embedding_time:add(nn.CAddTable());


	-- Define the inner product + softmax between input and question:
	inner_prod = nn.Sequential();
	PT = nn.ParallelTable();
	PT:add(question_embedding);
	PT:add(sent_input_embedding_time);
	inner_prod:add(PT);
	inner_prod:add(nn.MM(false, true));
	inner_prod:add(nn.SoftMax());

	-- Define the weighted sum:
	weighted_sum = nn.MM();

	-- Define the part of the model that yields the o vector using a weighted sum:
	model_inner = nn.Sequential();
	model_pt_inner = nn.ParallelTable();
	model_pt_inner:add(inner_prod);
	model_pt_inner:add(sent_output_embedding_time);

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

function accuracy(sentences, questions, questions_sentences, answers, model, memsize, nvoc, nans)

	local acc_task = torch.zeros(sentences:narrow(2,1,1):max())
	local task_count = torch.zeros(sentences:narrow(2,1,1):max())
	local acc = 0
	local memsize_range = torch.linspace(1,memsize,memsize):type('torch.LongTensor')
	local memory = torch.ones(memsize, sentences:size(2)-1)*nvoc
	for i = 1, questions:size(1) do
		-- xlua.progress(i, questions:size(1))
		memory:fill(nvoc)
        story = sentences:narrow(2,2,sentences:size(2)-1):narrow(1,questions_sentences[i][1], questions_sentences[i][2]-questions_sentences[i][1]+1)
        
        -- ONLY TAKES UP TO THE LAST memsize FACTS:
		if story:size(1) < memsize then 
        	memory:narrow(1,memsize-story:size(1)+1,story:size(1)):copy(story)
        else
        	memory:copy(story:narrow(1, story:size(1) - memsize + 1, memsize))
        end 

        q = questions:narrow(2,2,questions:size(2)-1)[i]
        input = {{{q, {memory, memsize_range}}, {memory, memsize_range}}, q}

		pred = model:forward(input)
		m, a = pred:view(nans,1):max(1)

		if a[1][1] == answers[i][1] then
			acc = acc + 1
			acc_task[questions[i][1]] = acc_task[questions[i][1]] + 1.
		end
		task_count[questions[i][1]] = task_count[questions[i][1]] + 1.
	end
	return acc/questions:size(1), acc_task:cdiv(task_count)
end


function train_model(sentences, questions, questions_sentences, answers, model, nvoc, nans, memsize, criterion, eta, nEpochs)
    -- Train the model with a SGD
    -- standard parameters are
    -- nEpochs = 1
    -- eta = 0.01

    -- To store the loss
    local loss = torch.zeros(nEpochs)
    local accuracy_ = torch.zeros(nEpochs)
    local av_L = 0

    local memsize_range = torch.linspace(1,memsize,memsize):type('torch.LongTensor')
    local memory = torch.ones(memsize, sentences:size(2)-1)*nvoc

    for i = 1, nEpochs do
    	-- Display progess
        xlua.progress(i, nEpochs)

        -- timing the epoch
        timer = torch.Timer()
        av_L = 0
        if i % 25 == 0 and i < 100 then
        	eta = eta/2
        end
        -- mini batch loop
        for t = 1, questions:size(1) do
            -- define input:
            memory:fill(nvoc)

            story = sentences:narrow(2,2,sentences:size(2)-1):narrow(1,questions_sentences[t][1], questions_sentences[t][2]-questions_sentences[t][1]+1)
            -- ONLY TAKES UP TO THE LAST memsize FACTS:
            if story:size(1) < memsize then 
            	memory:narrow(1,memsize-story:size(1)+1,story:size(1)):copy(story)
            else
            	memory:copy(story:narrow(1, story:size(1) - memsize + 1, memsize))
            end
            
            q = questions:narrow(2,2,questions:size(2)-1)[t]
            input = {{{q, {memory, memsize_range}}, {memory, memsize_range}}, q}

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

        accuracy_[i], task_acc = accuracy(sentences, questions, questions_sentences, answers, model, memsize, nvoc, nans)
        loss[i] = av_L/questions:size(1)
        print('Epoch '..i..': '..timer:time().real)
       	print('\n')
        print('Average Loss: '.. loss[i])
        print('\n')
        print('Training accuracy: '.. accuracy_[i])
        print('\n')
        print('***************************************************')
       	
    end

    return loss, accuracy, task_acc
end

-- Sanity check:

myFile = hdf5.open('../Data/preprocess/all_train.hdf5','r')
f = myFile:all()
sentences = f['sentences']
questions = f['questions']
questions_sentences = f['questions_sentences']
answers = f['answers']
nvoc = f['voc_size'][1]
myFile:close()
nans = answers:max()

-- Building the model
hid = 50
memsize = 50
model = buildmodel(hid, nvoc, nans, memsize)

-- Initialise parameters using normal(0,0.1) as mentioned in the paper
parameters, gradParameters = model:getParameters()
randomkit.normal(parameters, 0, 0.1)

-- Criterion
criterion = nn.ClassNLLCriterion()

-- Training
eta = 0.01
nEpochs = 50
loss_train, accuracy_train, accuracy_train_task = train_model(sentences, questions, questions_sentences, answers, model, nvoc, nans, memsize, criterion, eta, nEpochs)
print('Detailed accuracies:')
print(accuracy_train_task)

