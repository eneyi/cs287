require 'nngraph';

-- Parameters
memsize = 50
dim_hidden = 50
num_answer = 6
voca_size = 10

----------------------------------------------------
-- Model
----------------------------------------------------

-- Debugging
nngraph.setDebug(true)

-- Inputs
story_in_memory = nn.Identity()()
question = nn.Identity()()
time = nn.Identity()()

-- Adjacent model

A = nn.LookupTable(voca_size, dim_hidden)
T_A = nn.LookupTable(memsize, dim_hidden)
B = nn.LookupTable(voca_size, dim_hidden)
-- Set B = A
B:parameters()[1] = A:parameters()[1]
C = nn.LookupTable(voca_size, dim_hidden)
T_C = nn.LookupTable(memsize, dim_hidden)

-- Hop 1
question_embedding = nn.View(1, dim_hidden)(nn.Sum(1)(B(question)));
sent_input_embedding = nn.CAddTable()({nn.Sum(2)(A(story_in_memory)), T_A(time)});
sent_output_embedding = nn.CAddTable()({nn.Sum(2)(C(story_in_memory)), T_C(time)});

weights = nn.SoftMax()(nn.MM(false, true)({question_embedding, sent_input_embedding}))
o = nn.MM()({weights, sent_output_embedding})

-- -- Hop 2 (to comment out to see the bug)

-- set A=C; T_A = T_C
A = nn.LookupTable(voca_size, dim_hidden)
T_A = nn.LookupTable(memsize, dim_hidden)
A:share(C,'weight', 'gradWeight', 'bias', 'gradBias')
T_A:share(T_C,'weight', 'gradWeight', 'bias', 'gradBias')

-- New C
C = nn.LookupTable(voca_size, dim_hidden)
T_C = nn.LookupTable(memsize, dim_hidden)

question_embedding = nn.CAddTable()({o, question_embedding})
sent_input_embedding = nn.CAddTable()({nn.Sum(2)(A(story_in_memory)), T_A(time)});
sent_output_embedding = nn.CAddTable()({nn.Sum(2)(C(story_in_memory)), T_C(time)});

weights = nn.SoftMax()(nn.MM(false, true)({question_embedding, sent_input_embedding}))
o = nn.MM()({weights, sent_output_embedding})

-- Response
question_embedding = nn.CAddTable()({o, question_embedding})
W = nn.Linear(dim_hidden, num_answer, false)
output = nn.LogSoftMax()(W(question_embedding))

-- Model
-- Debugging
nngraph.annotateNodes()

model = nn.gModule({story_in_memory, question, time}, {output})
model.name = 'buggy'
graph.dot(model.fg, 'MLP', 'multiple_hops')

----------------------------------------------------
-- Forward
----------------------------------------------------

-- Synthetic data
story_memory = torch.ones(memsize, 6)
question_input = torch.ones(5)*3
time_input = torch.linspace(1,memsize,memsize):type('torch.LongTensor')
answers = torch.ones(1)

model_output = model:updateOutput({story_memory, question_input, time_input})
print('Output with NNgraph')
print(model_output)

----------------------------------------------------
-- Backward
----------------------------------------------------
criterion = nn.ClassNLLCriterion()
df_do = criterion:backward(model_output, answers)
print(df_do)
-- Uncomment next line to raise the error
back_o = model:updateGradInput({story_input, question_input, time_input}, df_do)
-- Uncomment next line to catch the error and save the graph plot with the error node
-- pcall( function() back_o = model:updateGradInput({story_input, question_input, time_input}, df_do) end)
print(back_o)
model:updateParameters(0.1)