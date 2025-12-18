import os
from collections import OrderedDict

import torch


def early_stopping(recall_list, stopping_steps):
    best_recall = max(recall_list)                # 最优指标
    best_step = recall_list.index(best_recall)    # 最优指标的idx
    if len(recall_list) - best_step - 1 >= stopping_steps:
        should_stop = True
    else:
        should_stop = False
    return best_recall, should_stop


def save_model(model, model_dir, log_save_id, current_epoch, last_best_epoch=None):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_state_file = os.path.join(model_dir, 'model{:d}_epoch{}.pth'.format(log_save_id, current_epoch))
    torch.save({'model_state_dict': model.state_dict(), 'epoch': current_epoch}, model_state_file)

    if last_best_epoch is not None and current_epoch != last_best_epoch:    # 之前有过best_model且当前这个epoch不是best_model的epoch，则要把last_best_model删除。
        old_model_state_file = os.path.join(model_dir, 'model{:d}_epoch{}.pth'.format(log_save_id, last_best_epoch))
        if os.path.exists(old_model_state_file):
            os.system('rm {}'.format(old_model_state_file))


def load_model(model, model_path):
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


