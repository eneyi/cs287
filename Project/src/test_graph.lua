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
question_input = questions:narrow(1,i,1)
time_input = torch.linspace(1,memsize,memsize):type('torch.LongTensor')
story_memory = torch.ones(memsize, sentences:size(2)-1)*voca_size
story_memory:narrow(1,memsize - story_input:size(1)+1,story_input:size(1)):copy(story_input)

print(story_input)
print(question_input)
print(story_memory)

-- Parameters
dim_hidden = 50
num_answer = torch.max(answers)
print(voca_size)
print(num_answer)
print(dim_hidden)

-- Inputs
story_in_memory = nn.Identity()()
question = nn.Identity()()
time = nn.Identity()()

-- L1 = nn.Sum(2)(nn.LookupTable(voca_size, dim_hidden)(story_in_memory))
-- L2 = nn.LookupTable(memsize, dim_hidden)(time)
-- Embedding
question_embedding = nn.View(1, dim_hidden)(nn.Sum(2)(nn.LookupTable(voca_size, dim_hidden)(question)));
sent_input_embedding = (nn.CAddTable()({nn.Sum(2)(nn.LookupTable(voca_size, dim_hidden)(story_in_memory)),
                                       nn.LookupTable(memsize, dim_hidden)(time)}));
sent_output_embedding = nn.CAddTable()({nn.Sum(2)(nn.LookupTable(voca_size, dim_hidden)(story_in_memory)),
                                       nn.LookupTable(memsize, dim_hidden)(time)});

-- Components
weights = nn.SoftMax()(nn.MM(false, true)({question_embedding, sent_input_embedding}))
o = nn.MM()({weights, sent_output_embedding})
output = nn.SoftMax()(nn.Linear(dim_hidden, num_answer)(nn.Sum(1)(nn.JoinTable(1)({o, question_embedding}))))

-- Model
model = nn.gModule({story_in_memory, question, time}, {output})

-- Initialize parameters
parameters, gradParameters = model:getParameters()
print(parameters:size())
torch.manualSeed(0)
randomkit.normal(parameters, 0, 0.1)
-- print(parameters:narrow(1,1,10))

-- print(story_memory:size())
-- print(story_memory:max())
-- print(time_input:size())
-- print(time_input:max())
-- model_test = nn.gModule({story_in_memory, time}, {sent_input_embedding})
-- model_output = model_test:updateOutput({story_memory, time_input})
-- print(model_output:size())

print(story_input:size(), question_input:size())
model_output = model:updateOutput({story_memory, question_input, time_input})
print(model_output)

-- backward
-- criterion = nn.ClassNLLCriterion()
-- df_do = criterion:backward(model_output, answers[i])
-- print(df_do)
-- back_o = model:updateGradInput({story_input, question_input}, df_do)
-- print(back_o)
-- model:updateParameters(0.1)



function buildmodel(hid, nvoc, nans, memsize)

    -- Initialise the 3 lookup tables:
    local question_embedding = nn.Sequential();
    question_embedding:add(nn.LookupTable(nvoc, hid));
    question_embedding:add(nn.Sum(1));
    question_embedding:add(nn.View(1, hid));

    local sent_input_embedding_time = nn.Sequential();
    local sent_input_pt = nn.ParallelTable();
    local sent_input_embedding = nn.Sequential();
    sent_input_embedding:add(nn.LookupTable(nvoc, hid));
    sent_input_embedding:add(nn.Sum(2));
    local TA = nn.LookupTable(memsize,hid);
    sent_input_pt:add(sent_input_embedding);
    sent_input_pt:add(TA);
    sent_input_embedding_time:add(sent_input_pt);
    sent_input_embedding_time:add(nn.CAddTable());


    local sent_output_embedding_time = nn.Sequential();
    local sent_output_pt = nn.ParallelTable();
    local sent_output_embedding = nn.Sequential();
    sent_output_embedding:add(nn.LookupTable(nvoc, hid));
    sent_output_embedding:add(nn.Sum(2));
    local TC = nn.LookupTable(memsize,hid);
    sent_output_pt:add(sent_output_embedding);
    sent_output_pt:add(TC);
    sent_output_embedding_time:add(sent_output_pt);
    sent_output_embedding_time:add(nn.CAddTable());


    -- Define the inner product + softmax between input and question:
    local inner_prod = nn.Sequential();
    local PT = nn.ParallelTable();
    PT:add(question_embedding);
    PT:add(sent_input_embedding_time);
    inner_prod:add(PT);
    inner_prod:add(nn.MM(false, true));
    inner_prod:add(nn.SoftMax());

    -- Define the weighted sum:
    local weighted_sum = nn.MM();

    -- Define the part of the model that yields the o vector using a weighted sum:
    local model_inner = nn.Sequential();
    local model_pt_inner = nn.ParallelTable();
    model_pt_inner:add(inner_prod);
    model_pt_inner:add(sent_output_embedding_time);

    model_inner:add(model_pt_inner);
    model_inner:add(weighted_sum);

    -- Building the model itself:
    local model = nn.Sequential();
    local model_pt = nn.ParallelTable();

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

memsize = 50
model = buildmodel(dim_hidden, voca_size, num_answer, memsize)

-- Initialise parameters using normal(0,0.1) as mentioned in the paper
parameters, gradParameters = model:getParameters()
torch.manualSeed(0)
randomkit.normal(parameters, 0, 0.1)


input = {{{question_input, {story_memory, time_input}}, {story_memory, time_input}}, question_input}
pred = model:forward(input)
print(pred)
