import os
import random
import time

import torch
import numpy as np

from models.model import Model
from utils import load_adj, EHRDataset, format_time, MultiStepLRScheduler, FocalLoss
from metrics import evaluate_codes, evaluate_hf


def historical_hot(code_x, code_num, lens):
    result = np.zeros((len(code_x), code_num), dtype=int)
    for i, (x, l) in enumerate(zip(code_x, lens)):
        result[i] = x[l - 1]
    return result


if __name__ == '__main__':
    seed = 6669
    dataset = 'mimic3'  # 'mimic3' or 'eicu'
    # task = 'diabetes'  # 'm' or 'h'
    task = 'diabetes'
    use_cuda = True
    device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else 'cpu')

    code_size = 64
    graph_size = 256
    hidden_size = 150  # rnn hidden size
    t_attention_size = 32
    t_output_size = hidden_size
    batch_size = 64
    epochs = 10

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    dataset_path = os.path.join('data', dataset, 'standard')
    train_path = os.path.join(dataset_path, 'train')
    valid_path = os.path.join(dataset_path, 'valid')
    test_path = os.path.join(dataset_path, 'test')

    code_adj = load_adj(dataset_path, device=device)
    code_num = len(code_adj)
    print('loading train data ...')
    train_data = EHRDataset(train_path, label=task, batch_size=batch_size, shuffle=True, device=device)
    print('loading valid data ...')
    valid_data = EHRDataset(valid_path, label=task, batch_size=batch_size, shuffle=False, device=device)
    print('loading test data ...')
    test_data = EHRDataset(test_path, label=task, batch_size=batch_size, shuffle=False, device=device)

    test_historical = historical_hot(valid_data.code_x, code_num, valid_data.visit_lens)

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
            'evaluate_fn': evaluate_hf,
            'lr': {
                'init_lr': 0.001,
                'milestones': [2, 3, 20],
                'lrs': [1e-3, 1e-4, 1e-5]
            }
        },

        'diabetes': {
            'dropout': 0.2,
            'output_size': 1,
            'evaluate_fn': evaluate_hf,
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
    dropout_rate = task_conf[task]['dropout']

    param_path = os.path.join('data', 'params', dataset, task)
    if not os.path.exists(param_path):
        os.makedirs(param_path)

    model = Model(code_num=code_num, code_size=code_size,
                  adj=code_adj, graph_size=graph_size, hidden_size=hidden_size, t_attention_size=t_attention_size,
                  t_output_size=t_output_size,
                  output_size=output_size, dropout_rate=dropout_rate, activation=activation).to(device)
    
    lr = 0.01
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)
    # scheduler = MultiStepLRScheduler(optimizer, epochs, task_conf[task]['lr']['init_lr'],
    #                                  task_conf[task]['lr']['milestones'], task_conf[task]['lr']['lrs'])
    # scheduler = 
    # scheduler = 

    # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=lr,cycle_momentum= False, step_size_up = 80)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',factor=0.25,patience=2,verbose=True)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)

    best_val_f1 = -999
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
            code_x, visit_lens, divided, y, neighbors = train_data[step]
            print("code_x",code_x,code_x.shape)
            print("visit_len",visit_lens,visit_lens.shape)
            print('divided',divided,divided.shape)
            print('neighbor',neighbors,neighbors.shape)
            print('y',y,y.shape)

            output = model(code_x, divided, neighbors, visit_lens).squeeze()
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
            
            if (batch_index +1) % 20 == 0:
                valid_loss, f1_score = evaluate_fn(model, valid_data, loss_fn, output_size, test_historical)
                scheduler.step(f1_score)

                if f1_score > best_val_f1:
                    torch.save(model.state_dict(), os.path.join(param_path, 'best.pt'))
                    best_val_f1 = f1_score
                    print('Saved checkpoint')
            batch_index += 1
        train_data.on_epoch_end()
        et = time.time()
        time_cost = format_time(et - st)
        print('\r    Step %d / %d, time cost: %s, loss: %.4f' % (steps, steps, time_cost, total_loss / total_num))
        valid_loss, f1_score = evaluate_fn(model, valid_data, loss_fn, output_size, test_historical)
        if f1_score > best_val_f1:
            torch.save(model.state_dict(), os.path.join(param_path, 'best.pt'))
            scheduler.step(f1_score)
            best_val_f1 = f1_score
            print('Saved checkpoint')

    model.load_state_dict(torch.load(os.path.join(param_path, 'best.pt')))
    model.eval()
    print('Evaluating on the test set')
    test_historical = historical_hot(test_data.code_x, code_num, test_data.visit_lens)

    evaluate_fn(model, test_data, loss_fn, output_size, test_historical)
        # torch.save(model.state_dict(), os.path.join(param_path, '%d.pt' % epoch))
