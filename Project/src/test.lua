require 'nngraph';
require 'hdf5';
require 'nn';
require 'randomkit'
require 'mem2.lua'

myFile = hdf5.open('../Data/preprocess/task1_train.hdf5','r')
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
num_hops = 1
sentence_size = sentences:size(2) - 1
print(voca_size)
print(num_answer)
print(dim_hidden)

batch_size = 2
cleaned_sentences = sentences:narrow(2,2,sentences:size(2)-1)
cleaned_questions = questions:narrow(2, 2, questions:size(2)-1)

time_input_batch = torch.linspace(1,memsize,memsize):type('torch.LongTensor'):repeatTensor(batch_size,1)
story_memory_batch = torch.ones(batch_size, memsize, sentences:size(2)-1)*voca_size
story_memory_batch_sized = torch.ones(batch_size, memsize * (sentences:size(2)-1))
question_input_batch = torch.zeros(batch_size, cleaned_questions:size(2))

build_input_batch(story_memory_batch, question_input_batch, cleaned_sentences, cleaned_questions, questions_sentences,
                   1, voca_size, batch_size)
print(story_memory_batch:size(), question_input_batch:size(), time_input_batch:size())

-- Graph model with multiple hops set with the Adjacent approach

----
--------------------- BATCH
----

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

question_embedding = nn.View(1, dim_hidden)(nn.Sum(2)(B(question)));

for K=1, num_hops do
    -- Initialization and A/T_A (next) = C/T_C (prev)
    A = nn.LookupTable(voca_size, dim_hidden)
    T_A = nn.LookupTable(memsize, dim_hidden)
    A:share(C,'weight', 'gradWeight', 'bias', 'gradBias')
    T_A:share(T_C,'weight', 'gradWeight', 'bias', 'gradBias')

    -- New C
    C = nn.LookupTable(voca_size, dim_hidden)
    T_C = nn.LookupTable(memsize, dim_hidden)

    -- Batch
    A_batched = nn.Sum(3)(nn.View(-1, memsize, sentence_size, dim_hidden)(A(story_in_memory)))
    -- C_batched = nn.Sum(3)(nn.View(-1, memsize, sentence_size, dim_hidden)(C(story_in_memory)))

    -- -- Transformed input
    sent_input_embedding = nn.CAddTable()({A_batched, T_A(time)});
    -- sent_output_embedding = nn.CAddTable()({C_batched, T_C(time)});

    -- -- Components
    -- weights = nn.SoftMax()(nn.MM(false, true)({question_embedding, sent_input_embedding}))
    weights = nn.MM(false, true)({question_embedding, sent_input_embedding})
    -- o = nn.MM()({weights, sent_output_embedding})

    -- -- Next step
    -- question_embedding = nn.CAddTable()({o, question_embedding})
end

-- W = nn.Linear(dim_hidden, num_answer, false)
-- -- TODO: set W^T = C (num_answer need to be size of voca_size)
-- -- W:parameters()[1] = W:parameters()[1]:transpose(1,2)

-- -- Final output
-- output = nn.LogSoftMax()(W(nn.View(-1, dim_hidden)(question_embedding)))

-- Model
-- model_batch = nn.gModule({story_in_memory, question, time}, {output})
model_batch = nn.gModule({story_in_memory, question, time}, {weights})


story_memory_batch_sized:copy(torch.view(story_memory_batch, batch_size, -1))
input = {story_memory_batch_sized, question_input_batch, time_input_batch}
-- input = {story_memory_batch_sized,time_input_batch}


parameters_batch, gradParameters_batch = model_batch:getParameters()
torch.manualSeed(0)
randomkit.normal(parameters_batch, 0, 0.1)
print('First 10 parameters ', parameters_batch:narrow(1, 1, 10))

print('input 1 is ', story_memory_batch:narrow(1,1,1))
print('input 2 is ', story_memory_batch:narrow(1,2,1))
print('question is ', question_input_batch)
pred = model_batch:forward(input)
-- print('pred is ', pred)
-- m, a = pred:max(2)
-- print('argmax pred ', a)

print('pred size', pred:size())
print('last line pred ', pred:narrow(1,1,1))
print('last line pred ', pred:narrow(1,2,1))
print(time_input_batch)

----
--------------------- NO BATCH
----


-- No batch
print('No Batch')
time_input = torch.linspace(1,memsize,memsize):type('torch.LongTensor')
story_memory = torch.ones(memsize, sentences:size(2)-1)*voca_size
question_input = torch.zeros(cleaned_questions:size(2))


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

    -- -- Transformed input
    sent_input_embedding = nn.CAddTable()({nn.Sum(2)(A(story_in_memory)), T_A(time)});
    -- sent_output_embedding = nn.CAddTable()({nn.Sum(2)(C(story_in_memory)), T_C(time)});

    -- -- Components
    -- weights = nn.SoftMax()(nn.MM(false, true)({question_embedding, sent_input_embedding}))
    weights = nn.MM(false, true)({question_embedding, sent_input_embedding})
    -- o = nn.MM()({weights, sent_output_embedding})

    -- -- Next step
    -- question_embedding = nn.CAddTable()({o, question_embedding})
end

-- W = nn.Linear(dim_hidden, num_answer, false)
-- -- TODO: set W^T = C (num_answer need to be size of voca_size)
-- -- W:parameters()[1] = W:parameters()[1]:transpose(1,2)

-- -- Final output
-- output = nn.LogSoftMax()(W(question_embedding))

-- Model

-- model = nn.gModule({story_in_memory, question, time}, {output})
model = nn.gModule({story_in_memory, question, time}, {weights})

parameters, gradParameters = model:getParameters()
parameters:copy(parameters_batch)
print('First 10 parameters ', parameters:narrow(1, 1, 10))

for i=1,batch_size do
    build_input(story_memory, question_input, cleaned_sentences, cleaned_questions, questions_sentences,
           i, voca_size)
    input = {story_memory, question_input, time_input}
    -- input = {story_memory, time_input}
    pred = model:forward(input)
    print('story is ', story_memory)
    print('question is ', question_input)
    -- print('pred is ', pred)
    -- m, a = pred:max(2)
    -- print('argmax pred ', a)
    print('pred size', pred:size())
    print('last line pred ', pred)

end

