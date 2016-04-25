require 'nngraph';
require 'hdf5';
require 'nn';


myFile = hdf5.open('../Data/preprocess/task2_train.hdf5','r')
f = myFile:all()
sentences = f['sentences']
questions = f['questions']
questions_sentences = f['questions_sentences']
answers = f['answers']
voca_size = f['voc_size'][1]
myFile:close()

i = 1
story_input = sentences:narrow(1,questions_sentences[i][1],
                               questions_sentences[i][2]-questions_sentences[i][1]+1)
question_input = questions:narrow(1,i,1)
print(story_input)
print(question_input)

-- Parameters
dim_hidden = 50
num_answer = torch.max(answers)
print(voca_size)
print(num_answer)
print(dim_hidden)

-- Inputs
story = nn.Identity()()
question = nn.Identity()()

-- Embedding
question_embedding = nn.View(1, dim_hidden)(nn.Sum(2)(nn.LookupTable(voca_size, dim_hidden)(question)));
sent_input_embedding = nn.Sum(2)(nn.LookupTable(voca_size, dim_hidden)(story));
sent_output_embedding = nn.Sum(2)(nn.LookupTable(voca_size, dim_hidden)(story));

-- Components
weights = nn.SoftMax()(nn.MM(false, true)({question_embedding, sent_input_embedding}))
o = nn.MM()({weights, sent_output_embedding})
output = nn.SoftMax()(nn.Linear(dim_hidden, num_answer)(nn.Sum(1)(nn.JoinTable(1)({o, question_embedding}))))

-- Model
model = nn.gModule({story, question}, {output})

print(story_input:size(), question_input:size())
model_output = model:updateOutput({story_input, question_input})
print(model_output)

-- backward
criterion = nn.ClassNLLCriterion()
df_do = criterion:backward(model_output, answers[i])
print(df_do)
back_o = model:updateGradInput({story_input, question_input}, df_do)
print(back_o)
model:updateParameters(0.1)