require 'hdf5';
require 'nngraph';
require 'torch';
require 'xlua';
require 'randomkit'

---------------------------------------------
---------------------------------------------
--
--          EXPERIMENTAL WORK
--
-- This is an experimental version of the Linear Start.
-- Not functional right now, the issue is that the model
-- without the softmax does not converge at all.


cmd = torch.CmdLine()

cmd:option('-filename','task1','prefix filename for preprocessed data')
cmd:option('-nepochs',10,'number of epochs')
cmd:option('-hops',3,'number of hops')
cmd:option('-mem',50,'Size of the memory')
cmd:option('-adjacent',1,'adjacent parameters if 1, else rnn like')
cmd:option('-tp',1,'Train proportion for the train/validation split')
cmd:option('-pe',0,'If 1, use of position encoding')
cmd:option('-ls',0,'If 1, use linear starting')
cmd:option('-extension',1,'Extension added to the file when saved')


function graph_model(dim_hidden, num_answer, voca_size, memsize, sentence_size)
    -- Inputs
    local story_in_memory = nn.Identity()()
    local question = nn.Identity()()
    local time = nn.Identity()()
    
    -- Position Encoding
    local sentence_size = sentence_size or 6
    local PE = torch.Tensor(memsize, sentence_size, dim_hidden)
    local PE_ = torch.Tensor(sentence_size, dim_hidden)

    for j = 1,sentence_size do
        for k = 1, dim_hidden do
            PE_[j][k] = (1-j/6)-(k/10)*(1-2*j/6)
        end
    end
    for i = 1, memsize do
        PE[i] = PE_
    end

    -- Embedding
    local question_embedding = nn.View(1, dim_hidden)(nn.Sum(1)(nn.LookupTable(voca_size, dim_hidden)(question)));
    local sent_input_embedding = nn.CAddTable()({nn.Sum(2)(nn.CMulTable()({nn.LookupTable(voca_size, dim_hidden)(story_in_memory),PE})),
                                           nn.LookupTable(memsize, dim_hidden)(time)});
    local sent_output_embedding = nn.CAddTable()({nn.Sum(2)(nn.CMulTable()({nn.LookupTable(voca_size, dim_hidden)(story_in_memory),PE})),
                                           nn.LookupTable(memsize, dim_hidden)(time)});

    -- Components
    local weights = nn.SoftMax()(nn.MM(false, true)({question_embedding, sent_input_embedding}))
    local o = nn.MM()({weights, sent_output_embedding})
    local output = nn.LogSoftMax()(nn.Linear(dim_hidden, num_answer, false)(nn.Sum(1)(nn.JoinTable(1)({o, question_embedding}))))

    -- Model
    local model = nn.gModule({story_in_memory, question, time}, {output})

    return model
end

function graph_model_hops_adjacent(dim_hidden, num_answer, voca_size, memsize, num_hops, PE, LS)
    -- Graph model with multiple hops set with the Adjacent approach
    -- parameter PE, if not nil use PE

    -- Inputs
    local story_in_memory = nn.Identity()()
    local question = nn.Identity()()
    local time = nn.Identity()()
    local PE_mem_node, PE_ques_node
    local question_embedding, A, T_A, sent_input_embedding, sent_output_embedding, weights, o
    if PE == 1 then
        PE_mem_node = nn.Identity()()
        PE_ques_node = nn.Identity()()
    end
  
    -- The initialization for C will serve for A
    local C = nn.LookupTable(voca_size, dim_hidden)
    local T_C = nn.LookupTable(memsize, dim_hidden)
    local B = nn.LookupTable(voca_size, dim_hidden)
    -- Set B = A (use of share to have them tied all along the train)
    B:share(C,'weight', 'gradWeight', 'bias', 'gradBias')

    if PE == 1 then
        question_embedding = nn.View(1, dim_hidden)(nn.Sum(1)(nn.CMulTable()({B(question),PE_ques_node})));
    else
        question_embedding = nn.View(1, dim_hidden)(nn.Sum(1)(B(question)));
    end

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
        if PE == 1 then
            sent_input_embedding = nn.CAddTable()({nn.Sum(2)(nn.CMulTable()({A(story_in_memory), PE_mem_node})), T_A(time)});
            sent_output_embedding = nn.CAddTable()({nn.Sum(2)(nn.CMulTable()({C(story_in_memory), PE_mem_node})), T_C(time)});
        else
            sent_input_embedding = nn.CAddTable()({nn.Sum(2)(A(story_in_memory)), T_A(time)});
            sent_output_embedding = nn.CAddTable()({nn.Sum(2)(C(story_in_memory)), T_C(time)});
        end

        -- Components
        if LS == 1 then
            -- we remove the softmax to keep the linearity
            weights = nn.MM(false, true)({question_embedding, sent_input_embedding})
        else
            weights = nn.SoftMax()(nn.MM(false, true)({question_embedding, sent_input_embedding}))
        end
        o = nn.MM()({weights, sent_output_embedding})

        -- Next step
        question_embedding = nn.CAddTable()({o, question_embedding})
    end

    local W = nn.Linear(dim_hidden, num_answer, false)
    -- TODO: set W^T = C (num_answer need to be size of voca_size)
    -- W:parameters()[1] = W:parameters()[1]:transpose(1,2)

    -- Final output
    local output = nn.LogSoftMax()(W(question_embedding))

    -- Model
    local model
    if PE == 1 then
        model = nn.gModule({story_in_memory, question, time, PE_mem_node, PE_ques_node}, {output})
    else
        model = nn.gModule({story_in_memory, question, time}, {output})
    end

    return model
end


function graph_model_hops_rnn_like(dim_hidden, num_answer, voca_size, memsize, num_hops, PE, LS)
    -- Inputs
    story_in_memory = nn.Identity()()
    question = nn.Identity()()
    time = nn.Identity()()
    if PE == 1 then
        PE_mem_node = nn.Identity()()
        PE_ques_node = nn.Identity()()
    end

    -- Initialization of embeddings
    A = nn.LookupTable(voca_size, dim_hidden)
    T_A = nn.LookupTable(memsize, dim_hidden)
    C = nn.LookupTable(voca_size, dim_hidden)
    T_C = nn.LookupTable(memsize, dim_hidden)
    B = nn.LookupTable(voca_size, dim_hidden)
    H = nn.Linear(dim_hidden, dim_hidden, false)

    if PE == 1 then
        question_embedding = nn.View(1, dim_hidden)(nn.Sum(1)(nn.CMulTable()({B(question),PE_ques_node})));
        sent_input_embedding = nn.CAddTable()({nn.Sum(2)(nn.CMulTable()({A(story_in_memory), PE_mem_node})), T_A(time)});
        sent_output_embedding = nn.CAddTable()({nn.Sum(2)(nn.CMulTable()({C(story_in_memory), PE_mem_node})), T_C(time)});
    else
        question_embedding = nn.View(1, dim_hidden)(nn.Sum(1)(B(question)));
        sent_input_embedding = nn.CAddTable()({nn.Sum(2)(A(story_in_memory)), T_A(time)});
        sent_output_embedding = nn.CAddTable()({nn.Sum(2)(C(story_in_memory)), T_C(time)});
    end

    -- Debugging
    nngraph.setDebug(true)

    for K=1, num_hops do
        -- Components
        if LS == 1 then
            -- we remove the softmax to keep the linearity
            weights = nn.MM(false, true)({question_embedding, sent_input_embedding})
        else
            weights = nn.SoftMax()(nn.MM(false, true)({question_embedding, sent_input_embedding}))
        end
        o = nn.MM()({weights, sent_output_embedding})

        -- Next step initialization
        question_embedding = nn.CAddTable()({o, H(question_embedding)})
    end

    W = nn.Linear(dim_hidden, num_answer, false)

    -- Final output
    output = nn.LogSoftMax()(W(question_embedding))

    -- Model
    if PE == 1 then
        model = nn.gModule({story_in_memory, question, time, PE_mem_node, PE_ques_node}, {output})
    else
        model = nn.gModule({story_in_memory, question, time}, {output})
    end

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

-- Compute the accuracies total and by task
function accuracy(sentences, questions, questions_sentences, answers, model, memsize, voca_size, dim_hidden, PE)
    -- To store per task metrics (row may be unused)
    local task_id_max = sentences:narrow(2, 1, 1):max() 
    local acc_by_task = torch.zeros(task_id_max, 2)
    -- Store index of the task
    acc_by_task:narrow(2, 1, 1):copy(torch.linspace(1, task_id_max, task_id_max))

    local count_task = torch.zeros(task_id_max)
    local acc = 0
    local time_input = torch.linspace(1,memsize,memsize):type('torch.LongTensor')
    local story_memory = torch.ones(memsize, sentences:size(2)-1)*voca_size
    -- Clean sentence and question while removing the task_id
    local cleaned_sentences = sentences:narrow(2,2,sentences:size(2)-1)
    local cleaned_questions = questions:narrow(2, 2, questions:size(2)-1)
    -- To store the quesiton input
    local question_input = torch.zeros(cleaned_questions:size(2))

    -- Build PE input if needed
    if PE == 1 then
        PE_mem, PE_ques = build_PE(cleaned_sentences:size(2), cleaned_questions:size(2),
                                   memsize, dim_hidden)     
    end
    for i = 1, questions:size(1) do
        -- Current task id
        t_id = questions[{i, 1}]

        -- Prediction
        build_input(story_memory, question_input, cleaned_sentences, cleaned_questions,
                    questions_sentences, i, voca_size)
        if PE == 1 then
            input = {story_memory, question_input, time_input, PE_mem, PE_ques}
        else
            input = {story_memory, question_input, time_input}
        end
        pred = model:forward(input)

        m, a = pred:view(pred:size(2),1):max(1)
        if a[1][1] == answers[i] then
            acc = acc + 1
            acc_by_task[{t_id, 2}] = acc_by_task[{t_id, 2}] + 1
        end
        count_task[t_id] = count_task[t_id] + 1
    end
    -- Normalize accuracy
    acc_by_task:narrow(2, 2, 1):cdiv(count_task)
    return acc/questions:size(1), acc_by_task
end

-- Compute only the total accuracy
function accuracy_total(sentences, questions, questions_sentences, time_input, answers, model, memsize, voca_size,
                  dim_hidden, PE)
    local acc = 0
    local story_memory = torch.ones(memsize, sentences:size(2)-1)*voca_size
    -- Clean sentence and question while removing the the task_id
    local cleaned_sentences = sentences:narrow(2,2,sentences:size(2)-1)
    local cleaned_questions = questions:narrow(2, 2, questions:size(2)-1)
    -- To store the question input
    local question_input = torch.zeros(cleaned_questions:size(2))

    -- Build PE input if needed
    if PE == 1 then
        PE_mem, PE_ques = build_PE(cleaned_sentences:size(2), cleaned_questions:size(2),
                                   memsize, dim_hidden)     
    end

    for i = 1, questions:size(1) do
        build_input(story_memory, question_input, cleaned_sentences, cleaned_questions, questions_sentences,
                   i, voca_size)
        if PE == 1 then
            input = {story_memory, question_input, time_input, PE_mem, PE_ques}
        else
            input = {story_memory, question_input, time_input}
        end
        pred = model:forward(input)
        -- print('pred is ', pred)

        m, a = pred:view(pred:size(2),1):max(1)
        if a[1][1] == answers[i] then
            acc = acc + 1
        end
    end
    return acc/questions:size(1)
end

function train_model(sentences, questions, questions_sentences, answers,
                    valid_questions, valid_questions_sentences, valid_answers,
                    model, par, gradPar, criterion, eta, nEpochs, memsize,
                    voca_size, valid_threshold, dim_hidden, num_answer, num_hops, PE, LS,
                    train_proportion, epsilon, adjacent)
    -- Train the model with a SGD

    -- To store the loss
    local loss = torch.zeros(nEpochs)
    local accuracy_tensor = torch.zeros(nEpochs)
    local accuracy_tensor_valid = torch.zeros(nEpochs)
    local av_L = 0
    local av_L_valid = 0

    local time_input = torch.linspace(1,memsize,memsize):type('torch.LongTensor')
    local story_memory = torch.ones(memsize, sentences:size(2)-1)*voca_size
    -- Clean sentence and question while removing the the task_id
    local cleaned_sentences = sentences:narrow(2,2,sentences:size(2)-1)
    local cleaned_questions = questions:narrow(2, 2, questions:size(2)-1)
    -- To store the quesiton input
    local question_input = torch.zeros(cleaned_questions:size(2))

    -- Build PE input if needed
    local PE_mem, PE_ques
    if PE == 1 then
        PE_mem, PE_ques = build_PE(cleaned_sentences:size(2), cleaned_questions:size(2),
                                   memsize, dim_hidden)     
    end

    -- Fix the initial eta if Linar Starting
    if LS == 1 then
        old_eta = eta
        eta = 0.005
        old_loss_valid = 10000000
        -- TO indicate if switched from linear to softmax
        switched = 0
    end

    ite = 1
    -- while loop used for LS
    while ite <= nEpochs do
        -- timing the epoch
        timer = torch.Timer()
        av_L = 0
        -- Updating eta only outside LS mode
        if LS == 0 or switched == 1 then
            if ite % 15 == 0 and ite < 100 then
                eta = eta/2
            end
        end
        -- mini batch loop
        print('eta used '.. eta)
        for i = 1, questions:size(1) do
            -- display progress
            xlua.progress(i, questions:size(1))

            build_input(story_memory, question_input, cleaned_sentences, cleaned_questions, questions_sentences,
                   i, voca_size)
            if PE == 1 then
                input = {story_memory, question_input, time_input, PE_mem, PE_ques}
            else
                input = {story_memory, question_input, time_input}
            end

            -- reset gradients
            model:zeroGradParameters()

            -- Forward pass
            pred = model:forward(input)
            -- Average loss computation
            f = criterion:forward(pred, answers[i])
            av_L = av_L + f

            -- Backward pass
            df_do = criterion:backward(pred, answers[i])
            model:backward(input, df_do)

            -- gradient normalization with max norm 40 (l2 norm)
            local gn = gradPar:norm()
            if gn > 40 then
                gradPar:mul(40 / gn)
            end
            model:updateParameters(eta)
            --par:add(gradPar:mul(-eta))
            
        end
        accuracy_tensor[ite] = accuracy_total(sentences, questions, questions_sentences, time_input, answers, model, memsize, voca_size,
                                            dim_hidden, PE)
        if train_proportion ~= 1 then
            accuracy_tensor_valid[ite] = accuracy_total(sentences, valid_questions, valid_questions_sentences, time_input, valid_answers,
                                                      model, memsize, voca_size, dim_hidden, PE)
        end
        loss[ite] = av_L/questions:size(1)
        print('Epoch '..ite..': '..timer:time().real)
        print('\n')
        print('Average Loss: '.. loss[ite])
        print('\n')
        print('Training accuracy: '.. accuracy_tensor[ite])
        print('\n')
        print('Validation accuracy: '.. accuracy_tensor_valid[ite])
        print('\n')
        print('***************************************************')

        -- Check valid loss if LS
        av_L_valid = 0
        if LS == 1 and switched == 0 then
            for i = 1, valid_questions:size(1) do
                build_input(story_memory, question_input, cleaned_sentences,
                            valid_questions:narrow(2, 2, valid_questions:size(2)-1),
                            valid_questions_sentences, i, voca_size)
                if PE == 1 then
                    input = {story_memory, question_input, time_input, PE_mem, PE_ques}
                else
                    input = {story_memory, question_input, time_input}
                end

                -- Prediction on valid
                pred_valid = model:forward(input)
                -- Average loss computation
                f = criterion:forward(pred_valid, valid_answers[i])
                print('valid question '..i)
                print('pred_valid ')
                print(pred_valid)
                print('valid_answers '.. valid_answers[i])
                print('loss '..f)
                av_L_valid = av_L_valid + f
            end
            av_L_valid = av_L_valid/valid_questions:size(1)
            print('Average valid loss ', av_L_valid)
            -- Check if valid loss still decreasing
            improvement_percentage = (av_L_valid - old_loss_valid)/av_L_valid
            if improvement_percentage < epsilon then
                old_loss_valid = av_L_valid
            else
                print('Switching from Linear to Softmax at epoch ' .. ite)
                -- Creating new model with SoftMax
                if adjacent == 1 then
                    new_model = graph_model_hops_adjacent(dim_hidden, num_answer, voca_size, memsize, num_hops, PE, 0)
                else
                    new_model = graph_model_hops_rnn_like(dim_hidden, num_answer, voca_size, memsize, num_hops, PE, 0)
                end
                -- Initializing parameters
                old_par = par:clone()
                par, gradPar = new_model:getParameters()
                -- randomkit.normal(par, 0, 0.1)
                -- Copying parameters
                par:copy(old_par)
                -- Changing model pointer
                model = new_model
                -- Restart training
                ite = 0
                -- Switch on
                switched = 1
                -- Reset eta
                eta = old_eta
            end
        end
        -- if accuracy_tensor_valid[i] > valid_threshold then
        --     return loss:narrow(1,1,i), accuracy_tensor:narrow(1,1,i), accuracy_tensor_valid:narrow(1,1,i)
        -- end

        -- Update
        ite = ite + 1
    end
    return loss, accuracy_tensor, accuracy_tensor_valid, model
end

--Position Encoding initialization
function build_PE(sentence_size, question_size, memsize, dim_hidden)     
    PE_mem = torch.Tensor(memsize, sentence_size, dim_hidden)

    for j = 1,sentence_size do
        for k = 1, dim_hidden do
            PE_mem:narrow(3, k, 1):narrow(2, j, 1):fill((1-j/sentence_size)-(k/dim_hidden)*(1-2*j/sentence_size))
        end
    end

    PE_ques = torch.Tensor(question_size, dim_hidden)
    for j = 1,question_size do
        for k = 1, dim_hidden do
            PE_ques[{j, k}] = (1-j/question_size)-(k/dim_hidden)*(1-2*j/question_size)
        end
    end

    return PE_mem, PE_ques
end

function main()
    -- Parsing arg
    opt = cmd:parse(arg)

    myFile = hdf5.open('../Data/preprocess/'.. opt.filename ..'_train.hdf5','r')
    f = myFile:all()
    sentences = f['sentences']
    questions = f['questions']
    questions_sentences = f['questions_sentences']
    answers = f['answers']
    voca_size = f['voc_size'][1]
    myFile:close()

    -- Train and validation:
    ndata = questions:size(1)
    perm = torch.randperm(ndata):long()
    train_proportion = opt.tp

    train_questions = questions:index(1,perm):narrow(1,1,math.floor(train_proportion*ndata))
    train_questions_sentences = questions_sentences:index(1,perm):narrow(1,1,math.floor(train_proportion*ndata))
    train_answers = answers:index(1,perm):narrow(1,1,math.floor(train_proportion*ndata))

    -- Check if split asked 
    if train_proportion ~= 1 then
        valid_questions = questions:index(1,perm):narrow(1,math.floor(train_proportion*ndata)+1,ndata-math.floor(train_proportion*ndata))
        valid_questions_sentences = questions_sentences:index(1,perm):narrow(1,math.floor(train_proportion*ndata)+1,ndata-math.floor(train_proportion*ndata))
        valid_answers = answers:index(1,perm):narrow(1,math.floor(train_proportion*ndata)+1,ndata-math.floor(train_proportion*ndata))
    end

    -- Parameters of the model
    memsize = opt.mem
    nEpochs = opt.nepochs
    PE = opt.pe
    LS = opt.ls
    adjacent = opt.adjacent
    eta = 0.01
    dim_hidden = 50
    num_hops = opt.hops
    num_answer = torch.max(answers)
    valid_threshold = 0.86
    epsilon = 0.05

    -- model = graph_model(dim_hidden, num_answer, voca_size, memsize)
    if adjacent == 1 then 
        model = graph_model_hops_adjacent(dim_hidden, num_answer, voca_size, memsize, num_hops, PE, LS)
    else
        model = graph_model_hops_rnn_like(dim_hidden, num_answer, voca_size, memsize, num_hops, PE, LS)
    end

    -- Initialise parameters using normal(0,0.1) as mentioned in the paper
    parameters, gradParameters = model:getParameters()
    randomkit.normal(parameters, 0, 0.1)

    -- -- Criterion
    criterion = nn.ClassNLLCriterion()

    -- -- Training
    loss_train, accuracy_tensor_train, accuracy_tensor_valid, model = train_model(sentences, train_questions,
                                                                           train_questions_sentences, train_answers,
                                                                           valid_questions, valid_questions_sentences,
                                                                           valid_answers ,model, parameters,
                                                                           gradParameters, criterion, eta, nEpochs,
                                                                           memsize, voca_size, valid_threshold,
                                                                           dim_hidden, num_answer, num_hops, PE, LS,
                                                                           train_proportion, epsilon, adjacent)

    accuracy_train, accuracy_by_task_train = accuracy(sentences, train_questions,
                                                      train_questions_sentences, train_answers,
                                                      model, memsize, voca_size, dim_hidden, PE)
    if train_proportion ~= 1 then 
        accuracy_valid, accuracy_by_task_valid = accuracy(sentences, valid_questions,
                                                      valid_questions_sentences, valid_answers,
                                                      model, memsize, voca_size, dim_hidden, PE)
    end

    print('Train accuracy TOTAL '.. accuracy_train)
    print('Train accuracy by task')
    print(accuracy_by_task_train)
    print('\n')
    print('***************************************************')

    if train_proportion ~= 1 then 
        print('Valid accuracy TOTAL '.. accuracy_valid)
        print('Valid accuracy by task')
        print(accuracy_by_task_valid)
        print('\n')
        print('***************************************************')
    end

    -- Prediction on test
    myFile = hdf5.open('../Data/preprocess/'.. opt.filename ..'_test.hdf5','r')
    f = myFile:all()
    sentences_test = f['sentences']
    questions_test = f['questions']
    questions_sentences_test = f['questions_sentences']
    answers_test = f['answers']
    voca_size = f['voc_size'][1]
    myFile:close()

    accuracy_test, accuracy_by_task_test = accuracy(sentences_test, questions_test,
                                                    questions_sentences_test, answers_test,
                                                    model, memsize, voca_size, dim_hidden, PE)
    print('Test accuracy TOTAL '.. accuracy_test)
    print('Test accuracy by task')
    print(accuracy_by_task_test)
    print('\n')
    print('***************************************************')

    -- Saving the final accuracies
    fname = 'accuracies/'..opt.filename .. '_' ..num_hops..'hops_'.. opt.adjacent..'adjacent_pe'.. PE ..'_ls'..LS..'_'.. opt.extension.. '_acc_by_task.hdf5'
    myFile = hdf5.open(fname, 'w')
    myFile:write('train', accuracy_by_task_train)
    myFile:write('test', accuracy_by_task_test)
    myFile:close()
    print('Accuracy by task saved at '.. fname)
end

main()