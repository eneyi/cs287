require 'nngraph';
require 'hdf5';
require 'nn';


myFile = hdf5.open('../Data/preprocess/task2_train.hdf5','r')
f = myFile:all()
sentences = f['sentences']
questions = f['questions']
questions_sentences = f['questions_sentences']
answers = f['answers']
myFile:close()


model = nn.Sequential()
model:add(nn.Identity())
model:add(nn.Tanh())

h1 = nn.Identity()()
h2 = nn.Identity()(nn.Tanh()(h1))
mlp = nn.gModule({h1}, {h2})

x = torch.rand(20)
print(x)

output_graph = mlp:forward(x)
print(output_graph)


output = model:forward(x)
print(output)

--graph.dot(mlp.fg, 'MLP', 'myMLP')
print(h1)
