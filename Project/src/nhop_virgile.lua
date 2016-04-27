require 'hdf5';
require 'nngraph';
require 'torch';
require 'xlua';
require 'randomkit'

-- function graph_model(hid, num_answer, nvoc, memsize)
--     print(hid, num_answer, nvoc)
--     -- Inputs
--     local story_in_memory = nn.Identity()()
--     local question = nn.Identity()()
--     local time = nn.Identity()()


--     -- Embedding
--     local question_embedding = nn.View(1, hid)(nn.Sum(1)(nn.LookupTable(nvoc, hid)(question)));
--     local sent_input_embedding = nn.CAddTable()({nn.Sum(2)(nn.LookupTable(nvoc, hid)(story_in_memory)),
--                                            nn.LookupTable(memsize, hid)(time)});
--     local sent_output_embedding = nn.CAddTable()({nn.Sum(2)(nn.LookupTable(nvoc, hid)(story_in_memory)),
--                                            nn.LookupTable(memsize, hid)(time)});

--     -- Components
--     local weights = nn.SoftMax()(nn.MM(false, true)({question_embedding, sent_input_embedding}))
--     local o = nn.MM()({weights, sent_output_embedding})



--     local output = nn.LogSoftMax()(nn.Linear(hid, num_answer, false)(nn.Sum(1)(nn.JoinTable(1)({o, question_embedding}))))

--     -- Model
--     local model = nn.gModule({story_in_memory, question, time}, {output})

--     return model
-- end

function graph_model(hid, nans, nvoc, memsize, nhop)

	-- Inputs
	local memory = nn.Identity()()
	local question = nn.Identity()()
	local time = nn.Identity()()

	local inner_res = {}

	--Embeddings
	local A_m = nn.Sum(2)(nn.LookupTable(nvoc, hid)(memory))
	local A_t = nn.LookupTable(memsize, hid)(time)
	local A = nn.CAddTable()({A_m,A_t})

	local C_m = nn.Sum(2)(nn.LookupTable(nvoc, hid)(memory))
	local C_t = nn.LookupTable(memsize, hid)(time)
	local C = nn.CAddTable()({C_m,C_t})

	local B = nn.View(1, hid)(nn.Sum(1)(nn.LookupTable(nvoc, hid)(question)))

	inner_res[1] = B

	for i = 2, nhop do
		local weights = nn.SoftMax()(nn.MM(false, true)({inner_res[i-1], A}))
		local o = nn.MM()({weights, C})
		if i ~= nhop then
			inner_res[i] = nn.Sum(1)(nn.JoinTable(1)({o, nn.Linear(hid, hid, false)(inner_res[i-1])}))
		else
			inner_res[i] = nn.Sum(1)(nn.JoinTable(1)({o, inner_res[i-1]}))
		end
	end

	local output = nn.LogSoftMax()(nn.Linear(hid, nans, false)(inner_res[nhop]))

	local model = nn.gModule({memory, question, time}, {output})

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


function train_model(sentences, questions, questions_sentences, answers, model, criterion, eta, nEpochs, memsize, nvoc)
    -- Train the model with a SGD
    -- standard parameters are
    -- nEpochs = 1
    -- eta = 0.01

    -- To store the loss
    loss = torch.zeros(nEpochs)
    local accuracy_ = torch.zeros(nEpochs)
    av_L = 0

    local time = torch.linspace(1,memsize,memsize):type('torch.LongTensor')
    local memory = torch.ones(memsize, sentences:size(2)-1)*nvoc

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
            -- xlua.progress(t, questions:size(1))
            -- define input:
            memory:fill(nvoc)
            local story = sentences:narrow(2,2,sentences:size(2)-1):narrow(1,questions_sentences[t][1],
                                                                         questions_sentences[t][2]-questions_sentences[t][1]+1)
            local question = questions:narrow(2,2,questions:size(2)-1)[t]
            if story:size(1) < memsize then 
                memory:narrow(1,memsize-story:size(1)+1,story:size(1)):copy(story)
            else
                memory:copy(story:narrow(1, story:size(1) - memsize + 1, memsize))
            end
            input = {memory, question, time}
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
            --par:add(gradPar:mul(-eta))
            
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
    return loss, accuracy_, task_acc
end



myFile = hdf5.open('../Data/preprocess/all_train.hdf5','r')
f = myFile:all()
sentences = f['sentences']
questions = f['questions']
questions_sentences = f['questions_sentences']
answers = f['answers']
nvoc = f['voc_size'][1]
myFile:close()

memsize = 50
hid = 50
nhop = 2
nans = torch.max(answers)

-- i = 1
-- story = sentences:narrow(2,2,sentences:size(2)-1):narrow(1,questions_sentences[i][1], questions_sentences[i][2]-questions_sentences[i][1]+1)
-- question = questions:narrow(2,2,questions:size(2)-1)[i]
-- time = torch.linspace(1,memsize,memsize):type('torch.LongTensor')
-- memory = torch.ones(memsize, sentences:size(2)-1)*nvoc
-- memory:narrow(1,memsize - story:size(1)+1,story:size(1)):copy(story)

-- model = graph_model(50, nans, nvoc, memsize, 2)
-- parameters, gradParameters = model:getParameters()
-- torch.manualSeed(0)
-- randomkit.normal(parameters, 0, 0.1)

-- print(model:forward({memory, question, time}):exp())

criterion = nn.ClassNLLCriterion()

loss_train, accuracy_train, task_acc = train_model(sentences, questions, questions_sentences, answers, model, criterion, 0.01, 100, memsize, nvoc)


