# -*- coding: utf-8 -*-
# @Time    : 2021/11/10 14:48
# @Author  : miliang
# @FileName: predict_utils.py
# @Software: PyCharm

import torch
from total_utils.train_utils import set_test


def test_predict(config,model_path, test_iter):
    model = torch.load(model_path, map_location=lambda storage, loc: storage.cuda(config.device))
    prf_s = set_test(config, model, test_iter, config.source_test_dir)
    metric_names = ['TI', 'TC', 'AI', 'AC']
    avg_score = 0
    for j, prf in enumerate(prf_s):
        avg_score += prf[2] * 100
        try:
            config.logger.info('{}: P:{:.1f}, R:{:.1f}, F:{:.1f}'.format(metric_names[j], prf[0] * 100, prf[1] * 100, prf[2] * 100))
        except:
            print('{}: P:{:.1f}, R:{:.1f}, F:{:.1f}'.format(metric_names[j], prf[0] * 100, prf[1] * 100, prf[2] * 100))



