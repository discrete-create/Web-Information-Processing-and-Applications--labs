import os
import sys
import random
from time import time

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

from model.KG_embedding_model import Embedding_based
from parser.parser_Embedding_based import *           
from utils.log_helper import *
from utils.metrics import kg_metrics_from_ranks       
from utils.model_helper import *
from data_loader.loader_kg import DataLoader        
from collections import defaultdict
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
from utils.metrics import kg_metrics_from_ranks


def evaluate_kg(model, dataloader, device, Ks=(1, 3, 10), eval_batch_size=128):
    """
    Filtered KG evaluation:
      - 对每个 (h, r, t_true)，对所有实体 t 计算 score
      - 将所有其它真实尾实体（来自 train+valid+test）过滤掉
      - 得到 filtered rank，再算 MR/MRR/Hits@K
    """
    model.eval()
    n_entities = dataloader.n_entities
    val_df: pd.DataFrame = dataloader.kg_valid_data

    # 构建 true_tail 字典
    true_tail = defaultdict(lambda: defaultdict(set))

    for df in [dataloader.kg_train_data,
               dataloader.kg_valid_data,
               dataloader.kg_test_data]:
        for h, r, t in zip(df['h'], df['r'], df['t']):
            true_tail[h][r].add(t)

    # 对验证集做 filtered ranking
    ranks = []

    with torch.no_grad():
        with tqdm(total=len(val_df), desc='Filtered KG Evaluation') as pbar:
            for start in range(0, len(val_df), eval_batch_size):
                end = min(start + eval_batch_size, len(val_df))
                batch = val_df.iloc[start:end]

                h_batch = torch.LongTensor(batch['h'].values).to(device)
                r_batch = torch.LongTensor(batch['r'].values).to(device)
                t_true_batch = torch.LongTensor(batch['t'].values).to(device)

                B = h_batch.size(0)

                for i in range(B):
                    h = int(h_batch[i].item())
                    r = int(r_batch[i].item())
                    t_true = int(t_true_batch[i].item())

                    # 所有候选 tail
                    t_all = torch.arange(n_entities, device=device)

                    # 计算 score（距离）
                    h_vec = torch.LongTensor([h]*n_entities).to(device)
                    r_vec = torch.LongTensor([r]*n_entities).to(device)

                    scores = model.calc_score(h_vec, r_vec, t_all)  # (n_entities,)
                    scores = scores.detach().cpu().numpy()
 
                    # 把除了当前 t_true 以外的所有真实 tail 全部屏蔽掉
                    for t in true_tail[h][r]:
                        if t != t_true:
                            scores[t] = np.inf      # 因为是距离，越小越好，+inf 表示排除

                    # 计算 rank
                    target_score = scores[t_true]
                    better = np.sum(scores < target_score)
                    rank = better + 1            # rank 从 1 开始
                    ranks.append(rank)

                    pbar.update(1)

    # 汇总指标
    metrics = kg_metrics_from_ranks(ranks, Ks=Ks)
    return metrics



def train(args):
    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # logging
    log_save_id = create_log_id(args.save_dir)
    logging_config(folder=args.save_dir, name='log{:d}'.format(log_save_id), no_console=False)
    logging.info(args)

    # GPU / CPU
    device = torch.device("cuda:" + str(args.gpu_id)) if args.cuda else torch.device("cpu")
    print("using device",device)
    # load data 
    data = DataLoader(args, logging)

    # construct model & optimizer
    model = Embedding_based(args, data.n_entities, data.n_relations)
    if args.use_pretrain == 1:
        model = load_model(model, args.pretrain_model_path)

    model.to(device)
    logging.info(model)

    optimizer = optim.Adagrad(
    model.parameters(),
    lr=args.lr
)

    #  metrics & early stopping
    best_epoch = -1
    best_mrr = -1.0

    Ks = tuple(eval(args.Ks))
    eval_batch_size = args.test_batch_size

    epoch_list = []
    # 记录每个 epoch 的 KG 指标
    kg_metrics_history = {
        'MR': [],
        'MRR': [],
    }
    for k in Ks:
        kg_metrics_history[f'Hits@{k}'] = []

    # train model 
    for epoch in range(1, args.n_epoch + 1):
        model.train()

        time1 = time()
        total_loss = 0.0

        n_batch = data.n_kg_data // data.kg_batch_size + 1

        for iter in range(1, n_batch + 1):
            time2 = time()

            # 从训练 KG 中采一个 batch 的 (h, r, pos_t, neg_t)
            kg_batch_head, kg_batch_relation, kg_batch_pos_tail, kg_batch_neg_tail = \
                data.generate_kg_batch(data.kg_dict, data.kg_batch_size, data.n_entities)

            kg_batch_head = kg_batch_head.to(device)
            kg_batch_relation = kg_batch_relation.to(device)
            kg_batch_pos_tail = kg_batch_pos_tail.to(device)
            kg_batch_neg_tail = kg_batch_neg_tail.to(device)

            batch_loss = model(kg_batch_head,
                               kg_batch_relation,
                               kg_batch_pos_tail,
                               kg_batch_neg_tail,
                               is_train=True)

            if np.isnan(batch_loss.cpu().detach().numpy()):
                logging.info('ERROR: Epoch {:04d} Iter {:04d} / {:04d} Loss is nan.'.format(
                    epoch, iter, n_batch))
                sys.exit()

            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += batch_loss.item()

            if (iter % args.print_every) == 0:
                logging.info(
                    'KG Training: Epoch {:04d} Iter {:04d} / {:04d} | Time {:.1f}s | '
                    'Iter Loss {:.4f} | Iter Mean Loss {:.4f}'.format(
                        epoch, iter, n_batch, time() - time2,
                        batch_loss.item(), total_loss / iter)
                )

        logging.info(
            'KG Training: Epoch {:04d} Total Iter {:04d} | Total Time {:.1f}s | '
            'Iter Mean Loss {:.4f}'.format(
                epoch, n_batch, time() - time1, total_loss / n_batch)
        )

        # ========= evaluate on validation KG =========
        if (epoch % args.evaluate_every) == 0 or epoch == args.n_epoch:
            time3 = time()
            metrics = evaluate_kg(model, data, device, Ks=Ks, eval_batch_size=eval_batch_size)

            log_str = 'KG Evaluation: Epoch {:04d} | Time {:.1f}s | MR {:.4f} | MRR {:.4f}'.format(
                epoch, time() - time3, metrics['MR'], metrics['MRR'])
            for k in Ks:
                log_str += ' | Hits@{} {:.4f}'.format(k, metrics[f'Hits@{k}'])
            logging.info(log_str)

            epoch_list.append(epoch)
            kg_metrics_history['MR'].append(metrics['MR'])
            kg_metrics_history['MRR'].append(metrics['MRR'])
            for k in Ks:
                kg_metrics_history[f'Hits@{k}'].append(metrics[f'Hits@{k}'])

            # 以 MRR 作为 early stopping 的指标
            cur_mrr_list = kg_metrics_history['MRR']
            best_mrr, should_stop = early_stopping(cur_mrr_list, args.stopping_steps)

            if should_stop:
                break

            # 如果当前 epoch 是目前为止 MRR 最好的，就保存模型
            if cur_mrr_list.index(best_mrr) == len(epoch_list) - 1:
                save_model(model, args.save_dir, log_save_id, epoch, best_epoch)
                logging.info('Save model on epoch {:04d}!'.format(epoch))
                best_epoch = epoch

    # save metrics to file
    metrics_df = {'epoch_idx': epoch_list}
    metrics_df['MR'] = kg_metrics_history['MR']
    metrics_df['MRR'] = kg_metrics_history['MRR']
    for k in Ks:
        metrics_df[f'Hits@{k}'] = kg_metrics_history[f'Hits@{k}']

    metrics_df = pd.DataFrame(metrics_df)
    metrics_df.to_csv(os.path.join(args.save_dir, 'kg_metrics.tsv'),
                      sep='\t', index=False)

    # print best metrics
    if best_epoch != -1:
        best_row = metrics_df.loc[metrics_df['epoch_idx'] == best_epoch].iloc[0].to_dict()
        log_str = (
            'Best KG Evaluation: Epoch {:04d} | MR {:.4f} | MRR {:.4f}'
            .format(int(best_row['epoch_idx']), best_row['MR'], best_row['MRR'])
        )
        for k in Ks:
            log_str += ' | Hits@{} {:.4f}'.format(k, best_row[f'Hits@{k}'])
        logging.info(log_str)
    else:
        logging.info('No best epoch found (maybe training stopped too early).')


def predict(args):

    # GPU / CPU
    device = torch.device("cuda:" + str(args.gpu_id)) if args.cuda else torch.device("cpu")

    # load data
    data = DataLoader(args, logging)

    # load model
    model = Embedding_based(args, data.n_entities, data.n_relations)
    model = load_model(model, args.pretrain_model_path)
    model.to(device)

    print("hello! I've loaded the KG model for you!")

    Ks = (1, 3, 10) if not hasattr(args, "Ks") else tuple(eval(args.Ks))
    eval_batch_size = getattr(args, "eval_batch_size", 128)

    metrics = evaluate_kg(model, data, device, Ks=Ks, eval_batch_size=eval_batch_size)
    print("KG Evaluation Results:")
    print("MR  = {:.4f}".format(metrics['MR']))
    print("MRR = {:.4f}".format(metrics['MRR']))
    for k in Ks:
        print("Hits@{} = {:.4f}".format(k, metrics[f'Hits@{k}']))


if __name__ == '__main__':

    args = parse_args()
    train(args)
    # predict(args)