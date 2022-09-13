# -*- coding: utf-8 -*-
# @Time    : 2021/10/21 17:12
# @Author  : miliang
# @FileName: train_utils.py
# @Software: PyCharm
import torch
from models.model_base import CasEE
import os
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
from collections import OrderedDict
import numpy as np
import json
from total_utils.metric import cal_scores_ti_tc_ai_ac


def train(config, train_iter, dev_iter):
    model = CasEE(config).to(config.device)
    # model.train()

    bert_params = list(map(id, model.pre_model.parameters()))
    other_params = filter(lambda p: id(p) not in bert_params, model.parameters())
    optimizer_grouped_parameters = [{'params': model.pre_model.parameters(), 'lr': config.lr_bert},
                                    {'params': other_params, 'lr': config.lr_task}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.lr_bert, correct_bias=False)

    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=len(train_iter) * config.train_epoch * config.warmup_prop,
                                                num_training_steps=len(train_iter) * config.train_epoch)
    best_score, best_epoch = 0, 1
    for i in range(1, config.train_epoch + 1):
        # with tqdm(total=len(train_iter), position=0, ncols=80, desc='训练中') as t:
        with tqdm(total=len(train_iter), position=0, dynamic_ncols=True, desc='训练中') as t:
            model.train()
            for batch_data in train_iter:
                batch_data = [data.to(config.device) for data in batch_data]
                input_ids, segment_ids, input_mask, data_type_id_s, type_vec_s, r_pos, t_m, t_s, t_e, a_s, a_e, a_m = batch_data
                model.zero_grad()
                loss, type_loss, trigger_loss, args_loss = model.forward(data_type_id_s, type_vec_s,
                                                                         input_ids, segment_ids, input_mask,
                                                                         r_pos, t_m, t_s, t_e, a_s, a_e, a_m)
                loss.backward()
                optimizer.step()
                scheduler.step()  # Update learning rate schedule

                t.set_description("Epoch {}".format(i))
                t.set_postfix(OrderedDict(
                    type_loss="{:.4}".format(type_loss.item() / len(input_ids)),
                    trigger_loss="{:.4}".format(trigger_loss.item() / len(input_ids)),
                    args_loss="{:.4}".format(args_loss.item() / len(input_ids)),
                    loss="{:.4}".format(loss.item() / len(input_ids))))
                t.update(1)

        prf_s = set_test(config, model, dev_iter, config.source_dev_dir)
        metric_names = ['TI', 'TC', 'AI', 'AC']
        avg_score = 0
        for j, prf in enumerate(prf_s):
            avg_score += prf[2] * 100
            config.logger.info('{}: P:{:.1f}, R:{:.1f}, F:{:.1f}'.format(metric_names[j], prf[0] * 100, prf[1] * 100, prf[2] * 100))
        avg_score = avg_score / 4

        if avg_score > best_score:
            best_score = avg_score
            best_epoch = i
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            output_model_file = os.path.join(os.path.join(config.model_save_path, 'best_model.bin'))
            torch.save(model_to_save, output_model_file)

        config.logger.info('current epoch is {},current score is {}。'.format(i, avg_score))
        config.logger.info('best epoch is {},best score is {}。'.format(best_epoch, best_score))

        # lr_scheduler学习率递减 step
        # print('dev set : step_{},precision_{}, recall_{}, f1_{}, loss_{}'.format(cum_step, p, r, f1, loss))
        # 保存模型


def gen_idx_event_dict(records):
    data_dict = {}
    for line in records:

        idx = line['id']
        events = line['events']
        data_dict[idx] = events
    return data_dict


def read_jsonl(fn):
    with open(fn, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    data = []
    for line in lines:
        data.append(json.loads(line))
    return data


def set_test(config, model, dev_iter, dev_file_path):
    pred_records = []
    with tqdm(total=len(dev_iter), position=0, dynamic_ncols=True, desc='验证中') as t:
        model.eval()
        for input_ids, segment_ids, input_mask, data_ids, content in dev_iter:
            input_ids = input_ids.to(config.device)
            segment_ids = segment_ids.to(config.device)
            input_mask = input_mask.to(config.device)

            predict_result = model.predict(input_ids, segment_ids, input_mask, data_ids, content)
            pred_records.append(predict_result)
            t.update(1)

    pred_dict = gen_idx_event_dict(pred_records)
    gold_records = read_jsonl(dev_file_path)
    gold_dict = gen_idx_event_dict(gold_records)
    prf_s = cal_scores_ti_tc_ai_ac(pred_dict, gold_dict)
    return prf_s
