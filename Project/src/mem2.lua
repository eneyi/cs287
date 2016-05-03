require 'hdf5';
require 'nngraph';
require 'torch';
-- require 'xlua';
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

function graph_model(dim_hidden, num_answer, voca_size, memsize)
    -- Inputs
    local story_in_memory = nn.Identity()()
    local question = nn.Identity()()
    local time = nn.Identity()()

    -- Embedding
    local question_embedding = nn.View(1, dim_hidden)(nn.Sum(1)(nn.LookupTable(voca_size, dim_hidden)(question)));
    local sent_input_embedding = nn.CAddTable()({nn.Sum(2)(nn.LookupTable(voca_size, dim_hidden)(story_in_memory)),
                                           nn.LookupTable(memsize, dim_hidden)(time)});
    local sent_output_embedding = nn.CAddTable()({nn.Sum(2)(nn.LookupTable(voca_size, dim_hidden)(story_in_memory)),
                                           nn.LookupTable(memsize, dim_hidden)(time)});

    -- Components
    local weights = nn.SoftMax()(nn.MM(false, true)({question_embedding, sent_input_embedding}))
    local o = nn.MM()({weights, sent_output_embedding})
    local output = nn.LogSoftMax()(nn.Linear(dim_hidden, num_answer, false)(nn.Sum(1)(nn.JoinTable(1)({o, question_embedding}))))

    -- Model
    local model = nn.gModule({story_in_memory, question, time}, {output})

    return model
end

function graph_model_hops_adjacent(dim_hidden, num_answer, voca_size, memsize, num_hops)
    -- Graph model with multiple hops set with the Adjacent approach

    -- Inputs
    story_in_memory = nn.Identity()()
    question = nn.Identity()()
    time = nn.Identity()()

    -- The initialization for C will serve for A
    C = nn.LookupTable(voca_size, dim_hidden)
    T_C = nn.LookupTable(memsize, dim_hidden)
    B = nn.LookupTable(voca_size, dim_hidden)
    -- Set B = A (use of share to have them tied all along the train)
    B:share(C,'weight', 'gradWeight', 'bias', 'gradBias')

    question_embedding = nn.View(1, dim_hidden)(nn.Sum(1)(B(question)));

    for K=1, num_hops do
        -- Initialization and A/T_A (next) = C/T_C (prev)
        A = nn.LookupTable(voca_size, dim_hidden)
        T_A = nn.LookupTable(memsize, dim_hidden)
        A:share(C,'weight', 'gradWeight', 'bias', 'gradBias')
        T_A:share(T_C,'weight', 'gradWeight', 'bias', 'gradBias')

        -- New C
        C = nn.LookupTable(voca_size, dim_hidden)
        T_C = nn.LookupTable(memsize, dim_hidden)

        -- Transformed input
        sent_input_embedding = nn.CAddTable()({nn.Sum(2)(A(story_in_memory)), T_A(time)});
        sent_output_embedding = nn.CAddTable()({nn.Sum(2)(C(story_in_memory)), T_C(time)});

        -- Components
        weights = nn.SoftMax()(nn.MM(false, true)({question_embedding, sent_input_embedding}))
        o = nn.MM()({weights, sent_output_embedding})

        -- Next step
        question_embedding = nn.CAddTable()({o, question_embedding})
    end

    W = nn.Linear(dim_hidden, num_answer, false)
    -- TODO: set W^T = C (num_answer need to be size of voca_size)
    -- W:parameters()[1] = W:parameters()[1]:transpose(1,2)

    -- Final output
    output = nn.LogSoftMax()(W(question_embedding))

    -- Model
    model = nn.gModule({story_in_memory, question, time}, {output})

    return model
end


function graph_model_hops_rnn_like(dim_hidden, num_answer, voca_size, memsize, num_hops)
    -- Inputs
    story_in_memory = nn.Identity()()
    question = nn.Identity()()
    time = nn.Identity()()

    -- Initialization of embeddings
    A = nn.LookupTable(voca_size, dim_hidden)
    T_A = nn.LookupTable(memsize, dim_hidden)
    C = nn.LookupTable(voca_size, dim_hidden)
    T_C = nn.LookupTable(memsize, dim_hidden)
    B = nn.LookupTable(voca_size, dim_hidden)
    H = nn.Linear(dim_hidden, dim_hidden, false)

    question_embedding = nn.View(1, dim_hidden)(nn.Sum(1)(B(question)));
    sent_input_embedding = nn.CAddTable()({nn.Sum(2)(A(story_in_memory)), T_A(time)});
    sent_output_embedding = nn.CAddTable()({nn.Sum(2)(C(story_in_memory)), T_C(time)});

    -- Debugging
    nngraph.setDebug(true)

    for K=1, num_hops do
        -- Components
        weights = nn.SoftMax()(nn.MM(false, true)({question_embedding, sent_input_embedding}))
        o = nn.MM()({weights, sent_output_embedding})

        -- Next step initialization
        question_embedding = nn.CAddTable()({o, H(question_embedding)})
    end

    W = nn.Linear(dim_hidden, num_answer, false)

    -- Final output
    output = nn.LogSoftMax()(W(question_embedding))

    -- Model
    model = nn.gModule({story_in_memory, question, time}, {output})

    return model
end

-- In place building of the input (to use pre-allocated memory)
function build_input(story_memory, question_input, cleaned_sentences, cleaned_questions, questions_sentences,
                   question_index, voca_size)
        -- Initialize story_memory with padding
        story_memory:fill(voca_size)
        -- Extract story and question
        local story_start = questions_sentences[{question_index,1}]
        local story_size = questions_sentences[{question_index,2}] - story_start + 1
        local story = cleaned_sentences:narrow(1,story_start, story_size)
        question_input:copy(cleaned_questions[question_index])
        
        -- Building input
        if story_size < memsize then 
            story_memory:narrow(1,memsize - story_size + 1,story_size):copy(story)
        else
            story_memory:copy(story:narrow(1, story_size - memsize + 1, memsize))
        end
end

function accuracy(sentences, questions, questions_sentences, answers, model, memsize, voca_size,
                  dim_hidden)
	local acc = 0
    local time_input = torch.linspace(1,memsize,memsize):type('torch.LongTensor')
    local story_memory = torch.ones(memsize, sentences:size(2)-1)*voca_size
    -- Clean sentence and question while removing the the task_id
    local cleaned_sentences = sentences:narrow(2,2,sentences:size(2)-1)
    local cleaned_questions = questions:narrow(2, 2, questions:size(2)-1)
    -- To store the quesiton input
    local question_input = torch.zeros(cleaned_questions:size(2))

	for i = 1, questions:size(1) do
        build_input(story_memory, question_input, cleaned_sentences, cleaned_questions, questions_sentences,
                   i, voca_size)
        input = {story_memory, question_input, time_input}
		pred = model:forward(input)
        -- print('pred is ', pred)

		m, a = pred:view(pred:size(2),1):max(1)
		if a[1][1] == answers[i][1] then
			acc = acc + 1
		end
	end
	return acc/questions:size(1)
end


function train_model(sentences, questions, questions_sentences, answers, model, par, gradPar,
                     criterion, eta, nEpochs, memsize, voca_size)
    -- Train the model with a SGD

    -- To store the loss
    local loss = torch.zeros(nEpochs)
    local accuracy_tensor = torch.zeros(nEpochs)
    local av_L = 0

    local time_input = torch.linspace(1,memsize,memsize):type('torch.LongTensor')
    local story_memory = torch.ones(memsize, sentences:size(2)-1)*voca_size
    -- Clean sentence and question while removing the the task_id
    local cleaned_sentences = sentences:narrow(2,2,sentences:size(2)-1)
    local cleaned_questions = questions:narrow(2, 2, questions:size(2)-1)
    -- To store the quesiton input
    local question_input = torch.zeros(cleaned_questions:size(2))

    for i = 1, nEpochs do
        -- timing the epoch
        timer = torch.Timer()
        av_L = 0
        if i % 25 == 0 and i < 100 then
        	eta = eta/2
        end
        -- mini batch loop
        for i = 1, questions:size(1) do
	        build_input(story_memory, question_input, cleaned_sentences, cleaned_questions, questions_sentences,
                   i, voca_size)
            input = {story_memory, question_input, time_input}

            -- reset gradients
            model:zeroGradParameters()

            -- Forward pass (selection of inputs_batch in case the batch is not full, ie last batch)
            pred = model:forward(input)
            -- Average loss computation
            f = criterion:forward(pred, answers[i])
            av_L = av_L +f

            -- Backward pass
            df_do = criterion:backward(pred, answers[i])
            model:backward(input, df_do)

            -- gradient normalization with max norm 40 (l2 norm)
            gradPar:view(gradPar:size(1),1):renorm(1,2,40)
            model:updateParameters(eta)
            --par:add(gradPar:mul(-eta))
            
        end
        accuracy_tensor[i] = accuracy(sentences, questions, questions_sentences, answers, model,
                                      memsize, voca_size)
        loss[i] = av_L/questions:size(1)
        print('Epoch '..i..': '..timer:time().real)
       	print('\n')
        print('Average Loss: '.. loss[i])
        print('\n')
        print('Training accuracy: '.. accuracy_tensor[i])
        print('\n')
        print('***************************************************')
       
    end
    return loss, accuracy_tensor
end


myFile = hdf5.open('../Data/preprocess/task123_train.hdf5','r')
f = myFile:all()
sentences = f['sentences']
questions = f['questions']
questions_sentences = f['questions_sentences']
answers = f['answers']
voca_size = f['voc_size'][1]
myFile:close()

-- Building the model
memsize = 50
nEpochs = 100
eta = 0.01
dim_hidden = 50
num_hops = 3
num_answer = torch.max(answers)
batch_size = 16
sentence_size = sentences:size(2) - 1
-- model = graph_model(dim_hidden, num_answer, voca_size, memsize)
model = graph_model_hops_adjacent(dim_hidden, num_answer, voca_size, memsize, num_hops)
-- model = graph_model_hops_rnn_like(dim_hidden, num_answer, voca_size, memsize, num_hops)

-- Initialise parameters using normal(0,0.1) as mentioned in the paper
parameters, gradParameters = model:getParameters()
torch.manualSeed(0)
randomkit.normal(parameters, 0, 0.1)

-- -- Criterion
criterion = nn.ClassNLLCriterion()

-- -- Training
-- loss_train, accuracy_train = train_model(sentences, questions, questions_sentences, answers,
--                                          model, parameters, gradParameters, criterion, eta,
--                                          nEpochs, memsize, voca_size)



