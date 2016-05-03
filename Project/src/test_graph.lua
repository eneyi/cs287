require 'nngraph';
require 'hdf5';
require 'nn';
require 'randomkit'


myFile = hdf5.open('../Data/preprocess/task2_train.hdf5','r')
f = myFile:all()
sentences = f['sentences']
questions = f['questions']
questions_sentences = f['questions_sentences']
answers = f['answers']
voca_size = f['voc_size'][1]
myFile:close()

i = 1
memsize = 50
story_input = sentences:narrow(2,2,sentences:size(2)-1):narrow(1,questions_sentences[i][1], questions_sentences[i][2]-questions_sentences[i][1]+1)
question_input = questions:narrow(2,2,questions:size(2)-1)[i]
time_input = torch.linspace(1,memsize,memsize):type('torch.LongTensor')
story_memory = torch.ones(memsize, sentences:size(2)-1)*voca_size
story_memory:narrow(1,memsize - story_input:size(1)+1,story_input:size(1)):copy(story_input)

print(story_input)
print(question_input)
print(story_memory)

-- Parameters
dim_hidden = 50
num_answer = torch.max(answers)
num_hops = 3
print(voca_size)
print(num_answer)
print(dim_hidden)



----- Batching
function build_input_batch(story_memory, question_input, cleaned_sentences, cleaned_questions, questions_sentences,
                   question_index, voca_size, batch_size)
        -- Dimensions
        -- story_memory (batch_size, memsize, max_len_sentence)
        -- question_input (batch_size, max_len_question)
        -- Initialize story_memory with padding
        story_memory:fill(voca_size)
        -- Extract story and question
        for s=1,batch_size do
            local story_start = questions_sentences[{question_index + s,1}]
            local story_size = questions_sentences[{question_index + s,2}] - story_start + 1
            local story = cleaned_sentences:narrow(1,story_start, story_size)
            -- Building input
            sm = story_memory:narrow(1, s, 1):view(story_memory:size(2), story_memory:size(3))
            if story_size < memsize then 
                sm:narrow(1,memsize - story_size + 1,story_size):copy(story)
            else
                sm:narrow(1, s, 1):copy(story:narrow(1, story_size - memsize + 1, memsize))
            end
        end
        question_input:copy(cleaned_questions:narrow(1, question_index, batch_size))
end

batch_size = 32
cleaned_sentences = sentences:narrow(2,2,sentences:size(2)-1)
cleaned_questions = questions:narrow(2, 2, questions:size(2)-1)

time_input_batch = torch.linspace(1,memsize,memsize):type('torch.LongTensor'):repeatTensor(batch_size,1)
story_memory_batch = torch.ones(batch_size, memsize, sentences:size(2)-1)*voca_size
question_input_batch = torch.zeros(batch_size, cleaned_questions:size(2))

build_input_batch(story_memory_batch, question_input_batch, cleaned_sentences, cleaned_questions, questions_sentences,
                   1, voca_size, batch_size)
print(story_memory_batch:size(), question_input_batch:size(), time_input_batch:size())

---------------------------------------------------------
---------------------------------------------------------

-- Inputs
story_in_memory = nn.Identity()()
question = nn.Identity()()
time = nn.Identity()()

-- Adjacent model
-- The initialization for C will serve for A
C = nn.LookupTable(voca_size, dim_hidden)
T_C = nn.LookupTable(memsize, dim_hidden)
B = nn.LookupTable(voca_size, dim_hidden)

-- Set B = A
B:share(C,'weight', 'gradWeight', 'bias', 'gradBias')

-- RNN like
-- A = nn.LookupTable(voca_size, dim_hidden)
-- T_A = nn.LookupTable(memsize, dim_hidden)
-- C = nn.LookupTable(voca_size, dim_hidden)
-- T_C = nn.LookupTable(memsize, dim_hidden)
-- B = nn.LookupTable(voca_size, dim_hidden)
-- H = nn.Linear(dim_hidden, dim_hidden, false)


question_embedding = nn.View(1, dim_hidden)(nn.Sum(2)(B(question)));

-- -- Debugging
nngraph.setDebug(true)

for K=1, num_hops do
    -- Initialization
    A = nn.LookupTable(voca_size, dim_hidden)
    T_A = nn.LookupTable(memsize, dim_hidden)
    A:share(C,'weight', 'gradWeight', 'bias', 'gradBias')
    T_A:share(T_C,'weight', 'gradWeight', 'bias', 'gradBias')

    -- New C
    C = nn.LookupTable(voca_size, dim_hidden)
    T_C = nn.LookupTable(memsize, dim_hidden)

    -- Batch
    A_batched = nn.Sum(3)(nn.View(-1, memsize, story_input:size(2), dim_hidden)(A(story_in_memory)))
    C_batched = nn.Sum(3)(nn.View(-1, memsize, story_input:size(2), dim_hidden)(A(story_in_memory)))

    -- Transformed input
    sent_input_embedding_batch = nn.CAddTable()({A_batched, T_A(time)});
    sent_output_embedding_batch = nn.CAddTable()({C_batched, T_C(time)});

    -- Components
    weights = nn.SoftMax()(nn.MM(false, true)({question_embedding, sent_input_embedding_batch}))
    o = nn.MM()({weights, sent_output_embedding_batch})

    -- Next step initialization
    -- RNN like
    -- question_embedding = nn.CAddTable()({o, H(question_embedding)})
    -- Adjacent
    question_embedding = nn.CAddTable()({o, question_embedding})
end

W = nn.Linear(dim_hidden, num_answer, false)
-- TODO: set W^T = C (num_answer need to be size of voca_size)
-- W:parameters()[1] = W:parameters()[1]:transpose(1,2)

-- Final output
output = nn.LogSoftMax()(W(nn.View(-1, dim_hidden)(question_embedding)))

-- Model
-- Debugging
nngraph.annotateNodes()

model = nn.gModule({story_in_memory, question, time}, {output})
-- model.name = 'buggy'
-- graph.dot(model.bg, 'MLP', 'multiple_hops_b')

-- Test

-- question_embedding = nn.View(1, dim_hidden)(nn.Sum(2)(B(question)));
-- A_batched = nn.Sum(3)(nn.View(batch_size, memsize, story_input:size(2), dim_hidden)(A(story_in_memory)))
-- C_batched = nn.Sum(3)(nn.View(batch_size, memsize, story_input:size(2), dim_hidden)(A(story_in_memory)))
-- sent_input_embedding_batch = nn.CAddTable()({A_batched, T_A(time)});
-- sent_output_embedding_batch = nn.CAddTable()({C_batched, T_C(time)});
-- weights = nn.SoftMax()(nn.MM(false, true)({question_embedding, sent_input_embedding_batch}))
-- o = nn.MM()({weights, sent_output_embedding_batch})
-- question_embedding = nn.CAddTable()({o, question_embedding})

-- W = nn.Linear(dim_hidden, num_answer, false)
-- output = nn.LogSoftMax()(W(nn.View(batch_size, dim_hidden)(question_embedding)))

-- C_batched = nn.LogSoftMax()(nn.Add(dim_hidden)(nn.Sum(2)(C(story_in_memory))))
-- C_batched = nn.Sum(3)(nn.View(batch_size, memsize, 6, dim_hidden)(C(story_in_memory)))
-- model_test = nn.gModule({story_in_memory, question, time}, {output})

-- story_memory_batch_test = story_memory_batch:view(batch_size, -1)
-- print(story_memory_batch_test:size())
-- model_output = model_test:forward({story_memory_batch_test, question_input_batch, time_input_batch})
-- print(model_output:size())

-- criterion = nn.ClassNLLCriterion()
-- print(answers:narrow(1,i,batch_size):view(batch_size):size())
-- df_do = criterion:backward(model_output, answers:narrow(1,i,batch_size):view(batch_size))
-- print(df_do)
-- -- pcall( function() back_o = model:backward({story_memory_batch, question_input_batch, time_input_batch}, df_do) end)
-- -- print(model_test:parameters()[1]:narrow(1,1,1))
-- back_o = model_test:backward({story_memory_batch_test, question_input_batch, time_input_batch}, df_do)
-- print(back_o)
-- model_test:updateParameters(0.1)
-- print(model_test:parameters()[1]:narrow(1,1,1))

-- print(B:forward(question_input):size())
-- print(C:forward(story_memory):size())


-- Initialize parameters
parameters, gradParameters = model:getParameters()
print(parameters:size())
torch.manualSeed(0)
randomkit.normal(parameters, 0, 0.1)

story_memory_batch_test = story_memory_batch:view(batch_size, -1)
model_output = model:updateOutput({story_memory_batch_test, question_input_batch, time_input_batch})
print('Output Batch')
print(model_output:size())
inp_size = 2
story_memory_solo = story_memory_batch:narrow(1,1,inp_size):view(inp_size, -1)
question_input_solo = question_input_batch:narrow(1,1,inp_size)
time_input_solo = time_input_batch:narrow(1,1,inp_size)
print(story_memory_solo:size())
print(time_input_solo:size())
print(question_input_solo:size())
model_output = model:updateOutput({story_memory_solo, question_input_solo, time_input_solo})
print('Output Solo')
print(model_output:size())
print(model_output:max(2))

-- -- -- backward
-- criterion = nn.ClassNLLCriterion()
-- print(answers:narrow(1,i,batch_size):view(batch_size):size())
-- df_do = criterion:backward(model_output, answers:narrow(1,i,batch_size):view(batch_size))
-- print(df_do)
-- -- pcall( function() back_o = model:backward({story_memory_batch, question_input_batch, time_input_batch}, df_do) end)
-- back_o = model:backward({story_memory_batch, question_input_batch, time_input_batch}, df_do)
-- print(back_o)
-- model:updateParameters(0.1)



------ To compare with NN

-- function buildmodel(hid, nvoc, nans, memsize)

--     -- Initialise the 3 lookup tables:
--     local question_embedding = nn.Sequential();
--     question_embedding:add(nn.LookupTable(nvoc, hid));
--     question_embedding:add(nn.Sum(1));
--     question_embedding:add(nn.View(1, hid));

--     local sent_input_embedding_time = nn.Sequential();
--     local sent_input_pt = nn.ParallelTable();
--     local sent_input_embedding = nn.Sequential();
--     sent_input_embedding:add(nn.LookupTable(nvoc, hid));
--     sent_input_embedding:add(nn.Sum(2));
--     local TA = nn.LookupTable(memsize,hid);
--     sent_input_pt:add(sent_input_embedding);
--     sent_input_pt:add(TA);
--     sent_input_embedding_time:add(sent_input_pt);
--     sent_input_embedding_time:add(nn.CAddTable());


--     local sent_output_embedding_time = nn.Sequential();
--     local sent_output_pt = nn.ParallelTable();
--     local sent_output_embedding = nn.Sequential();
--     sent_output_embedding:add(nn.LookupTable(nvoc, hid));
--     sent_output_embedding:add(nn.Sum(2));
--     local TC = nn.LookupTable(memsize,hid);
--     sent_output_pt:add(sent_output_embedding);
--     sent_output_pt:add(TC);
--     sent_output_embedding_time:add(sent_output_pt);
--     sent_output_embedding_time:add(nn.CAddTable());


--     -- Define the inner product + softmax between input and question:
--     local inner_prod = nn.Sequential();
--     local PT = nn.ParallelTable();
--     PT:add(question_embedding);
--     PT:add(sent_input_embedding_time);
--     inner_prod:add(PT);
--     inner_prod:add(nn.MM(false, true));
--     inner_prod:add(nn.SoftMax());

--     -- Define the weighted sum:
--     local weighted_sum = nn.MM();

--     -- Define the part of the model that yields the o vector using a weighted sum:
--     local model_inner = nn.Sequential();
--     local model_pt_inner = nn.ParallelTable();
--     model_pt_inner:add(inner_prod);
--     model_pt_inner:add(sent_output_embedding_time);

--     model_inner:add(model_pt_inner);
--     model_inner:add(weighted_sum);

--     -- Building the model itself:
--     local model = nn.Sequential();
--     local model_pt = nn.ParallelTable();

--     -- Adding the part leading to o:
--     model_pt:add(model_inner);
--     -- Adding the part leading to u:
--     model_pt:add(question_embedding);

--     model:add(model_pt);

--     -- Summing o and u:
--     model:add(nn.JoinTable(1));
--     model:add(nn.Sum(1));

--     -- Applying a linear transformation W without bias term
--     model:add(nn.Linear(hid, nans, false));

--     -- Applying a softmax function to obtain a distribution over the possible answers
--     model:add(nn.LogSoftMax());

--     return model
-- end

-- memsize = 50
-- model = buildmodel(dim_hidden, voca_size, num_answer, memsize)

-- -- Initialise parameters using normal(0,0.1) as mentioned in the paper
-- parameters, gradParameters = model:getParameters()
-- torch.manualSeed(0)
-- randomkit.normal(parameters, 0, 0.1)
-- print(parameters:size())


-- input = {{{question_input, {story_memory, time_input}}, {story_memory, time_input}}, question_input}
-- pred = model:forward(input)
-- print('Output with NN')
-- print(pred)
