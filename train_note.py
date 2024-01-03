import os
import random
import time

import torch
import numpy as np

from models.text_model.model import Model
from utils import load_adj, EHRDatasetNote, format_time, MultiStepLRScheduler, FocalLoss, load_prior
from metrics import evaluate_codes, evaluate_hf, evaluate_hf_note
import argparse
import pickle

from transformers import get_linear_schedule_with_warmup

def historical_hot(code_x, code_num, lens):
    result = np.zeros((len(code_x), code_num), dtype=int)
    for i, (x, l) in enumerate(zip(code_x, lens)):
        result[i] = x[l - 1]
    return result

def read_option():
    parser = argparse.ArgumentParser()
    # basic settings
    parser.add_argument('--seed', help='Random seed', type=int, default=42)
    parser.add_argument('--dataset', help='name of dataset', type=str, default='mimic3') #mimic3 or mimic4
    parser.add_argument('--task', help='name of prediction', type=str, default='diabetes') #diabetes or h
    parser.add_argument('--code_embedding_size', help='size of code embedding', type=int, default=64) 
    parser.add_argument('--graph_size', help='hidden size of graph', type=int, default=64) 
    parser.add_argument('--batch_size', help='batch size', type=int, default=4) 
    parser.add_argument('--epochs', help='Number of training epochs', type=int, default=10) 
    parser.add_argument('--lr', help='Learning rate', type=float, default=5e-5) 
    parser.add_argument('--weight_decay', help='Weight decay', type=float, default=1e-6) 
    parser.add_argument('--dropout', help='dropout', type=float, default=0.2) 
    parser.add_argument('--loss_fn', help='type of loss function', type=str, default='focal') 
    parser.add_argument('--result_save_path', help='path to save the test results', type=str, default='log/mimic3/diabetes/chet/') 
    parser.add_argument('--resume_training', help='resume training from previous checkpoint or not',  action="store_true", default=False)
    parser.add_argument('--eval_steps', help='Number of step per eval',  type=int, default=250)

    args = parser.parse_args()

    return args

if __name__ == '__main__':

    args = read_option()
    seed = args.seed
    dataset = args.dataset
    task = args.task
    
    code_size = args.code_embedding_size
    graph_size =args.graph_size
    hidden_size = 150
    t_attention_size = 32
    t_output_size =1
    batch_size = args.batch_size
    epochs = args.epochs
    use_cuda = True
    device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else 'cpu')
    with open("data/mimic3/standard/note_dictionary.pkl", "rb") as input_file:
        dictionary = pickle.load(input_file)
    # print(dictionary)
    vocab_size = len(dictionary.keys())
    # print(vocab_size)

    eval_steps = args.eval_steps
    # seed = 42
    # dataset = 'mimic4'  # 'mimic3' or 'eicu'
    # # task = 'diabetes'  # 'm' or 'h'
    # task = 'diabetes'

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    dataset_path = os.path.join('data', dataset, 'standard')
    train_path = os.path.join(dataset_path, 'train')
    valid_path = os.path.join(dataset_path, 'valid')
    test_path = os.path.join(dataset_path, 'test')

    result_save_path = args.result_save_path
    if not os.path.exists(result_save_path):
        os.makedirs(result_save_path)

    code_adj = load_adj(dataset_path, device=device)
    code_num = len(code_adj)
    print('code_num',code_num)

    if task == 'h':
        prior = load_prior('data/mimic3/standard/hf_prior.npz', device=device)
    elif task == 'diabetes':
        prior = load_prior('data/mimic3/standard/diabetes_prior.npz', device=device)
    
    m = prior.mean(0, keepdim=True)
    s = prior.std(0, unbiased=False, keepdim=True)
    prior -= m
    prior /= s    # print(torch.nonzero(prior), prior.shape)
    prior = prior/prior.sum()
    # prior = prior.unsqueeze(0).repeat(batch_size,1)
    print(prior, prior.shape)
    print('loading train data ...')
    train_data = EHRDatasetNote(train_path, label=task, batch_size=batch_size, shuffle=True, device=device)
    print('loading valid data ...')
    valid_data = EHRDatasetNote(valid_path, label=task, batch_size=batch_size, shuffle=False, device=device)
    print('loading test data ...')
    test_data = EHRDatasetNote(test_path, label=task, batch_size=batch_size, shuffle=False, device=device)

    valid_historical = historical_hot(valid_data.code_x, code_num, valid_data.visit_lens)

    print(len(train_data))

    task_conf = {
        'm': {
            'dropout': 0.45,
            'output_size': code_num,
            'evaluate_fn': evaluate_codes,
            'lr': {
                'init_lr': 0.01,
                'milestones': [20, 30],
                'lrs': [1e-3, 1e-5]
            }
        },
        'h': {
            'dropout': 0.2,
            'output_size': 1,
            'evaluate_fn': evaluate_hf_note,
            'lr': {
                'init_lr': 0.001,
                'milestones': [2, 3, 20],
                'lrs': [1e-3, 1e-4, 1e-5]
            }
        },

        'diabetes': {
            'dropout': 0.2,
            'output_size': 1,
            'evaluate_fn': evaluate_hf_note,
            'lr': {
                'init_lr': 0.001,
                'milestones': [2, 3, 20],
                'lrs': [1e-3, 1e-4, 1e-5]
            }
        }

    }
    output_size = task_conf[task]['output_size']
    activation = torch.nn.Sigmoid()
    # loss_fn = torch.nn.BCELoss()
    loss_fn = FocalLoss()
    evaluate_fn = task_conf[task]['evaluate_fn']
    dropout_rate = args.dropout

    param_path = os.path.join('data', 'params', dataset, task)
    if not os.path.exists(param_path):
        os.makedirs(param_path)

    model = Model(code_num=code_num, code_size=code_size,
                  adj=code_adj, graph_size=graph_size, hidden_size=hidden_size, t_attention_size=t_attention_size,
                  t_output_size=150,
                  output_size=output_size, dropout_rate=dropout_rate, activation=activation,
                  transformer_hidden_size = 384,
                  transformer_att_head_num = 2,text_embedding_size  = 256,vocab_size = vocab_size,
                  encoder_layers=4,transformer_dropout_rate=0.2).to(device)
    
    lr = args.lr
    weight_decay = args.weight_decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # scheduler = MultiStepLRScheduler(optimizer, epochs, task_conf[task]['lr']['init_lr'],
    #                                  task_conf[task]['lr']['milestones'], task_conf[task]['lr']['lrs'])
    # scheduler = 
    # scheduler = 

    # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=lr,cycle_momentum= False, step_size_up = 80)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',factor=0.25,patience=2,verbose=True)
    num_training_steps  = len(train_data) // batch_size * epochs
    num_warmup_steps = num_training_steps // 8
#     scheduler = get_linear_schedule_with_warmup(
#     optimizer, num_warmup_steps=0, 
#     num_training_steps=num_training_steps
# )
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)
    
    # if args.resume_training:
    #     model.load_state_dict(torch.load(os.path.join(param_path, 'best_bert.pt')))
    
    best_val_f1_auc = -999
    for epoch in range(epochs):
        print('Epoch %d / %d:' % (epoch + 1, epochs))
        model.train()
        total_loss = 0.0
        total_num = 0
        steps = len(train_data)
        st = time.time()
        batch_index = 0
        for step in range(len(train_data)):
            optimizer.zero_grad()

            code_x, visit_lens, divided, y, neighbors, note = train_data[step]
            note = note[:,:1024]
            mask_tensor = note != 0
            mask_tensor = mask_tensor.long()
            # print(mask_tensor, mask_tensor.shape)
            # note_mask
            # print(note, note.shape)
            output = model(code_x, divided, neighbors, visit_lens,note, mask_tensor).squeeze()
            # for name, param in model.named_parameters():
            #     if param.requires_grad:
            #         print(name, param.data.grad)

            loss = loss_fn(output, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            # scheduler.step()

            total_loss += loss.item() * output_size * len(code_x)
            total_num += len(code_x)

            end_time = time.time()
            remaining_time = format_time((end_time - st) / (step + 1) * (steps - step - 1))
            print('\r    Step %d / %d, remaining time: %s, loss: %.4f'
                  % (step + 1, steps, remaining_time, total_loss / total_num), end='')
            
            if (batch_index +1) % eval_steps == 0:
                print(batch_index)
                valid_loss, auc,f1_score,report = evaluate_fn(model, valid_data, loss_fn, output_size, valid_historical)
                auc_f1_avg = (f1_score + auc) / 2 
                scheduler.step(auc_f1_avg)
                
                if auc_f1_avg > best_val_f1_auc:
                    torch.save(model.state_dict(), os.path.join(param_path, 'best_bert_1.pt'))
                    print('Saved checkpoint')
                    best_val_f1_auc = auc_f1_avg
            batch_index += 1
        train_data.on_epoch_end()
        et = time.time()
        time_cost = format_time(et - st)
        print('\r    Step %d / %d, time cost: %s, loss: %.4f' % (steps, steps, time_cost, total_loss / total_num))
        valid_loss, auc,f1_score,report = evaluate_fn(model, valid_data, loss_fn, output_size, valid_historical)
        auc_f1_avg = (f1_score + auc) / 2 
        scheduler.step(auc_f1_avg)

        if auc_f1_avg > best_val_f1_auc:
            torch.save(model.state_dict(), os.path.join(param_path, 'best_bert_1.pt'))
            print('Saved checkpoint')
            best_val_f1_auc = auc_f1_avg

    model.load_state_dict(torch.load(os.path.join(param_path, 'best_bert_1.pt')))
    model.eval()
    print('Evaluating on the test set')
    test_historical = historical_hot(test_data.code_x, code_num, test_data.visit_lens)

    _,val_auc,val_f1,val_report = evaluate_fn(model, valid_data, loss_fn, output_size, valid_historical)
    print(val_auc, val_f1, val_report)
    _,test_auc,test_f1,test_report = evaluate_fn(model, test_data, loss_fn, output_size, test_historical)
    print(test_auc,test_f1,test_report)
        # torch.save(model.state_dict(), os.path.join(param_path, '%d.pt' % epoch))
