require 'hdf5';
require 'nn';
require 'torch';

myFile = hdf5.open('../Data/preprocess/task2_train.hdf5','r')
f = myFile:all()
sentences = f['sentences'] + 1
questions = f['questions'] + 1
questions_sentences = f['questions_sentences']
answers = f['answers']
myFile:close()

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

model = nn.Sequential();
model_pt = nn.ParallelTable();

model_inner = nn.Sequential();
model_pt_inner = nn.ParallelTable();
model_pt_inner:add(inner_prod);
model_pt_inner:add(sent_output_embedding);

model_inner:add(model_pt_inner);
model_inner:add(weighted_sum);

model_pt:add(model_inner);
model_pt:add(question_embedding);

model:add(model_pt);
model:add(nn.JoinTable(1));
model:add(nn.Sum(1));

model:add(nn.Linear(hid, nans, false));

model:add(nn.Softmax());

model:forward({{{questions[1],sentences:narrow(1,1,2)},sentences:narrow(1,1,2)},questions[1]})

tt = inner_prod:forward({questions[1],sentences:narrow(1,1,2)})  
