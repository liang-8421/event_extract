# -*- coding: utf-8 -*-
# @Time    : 2021/11/12 13:13
# @Author  : miliang
# @FileName: config.py
# @Software: PyCharm

import torch
from transformers.models.bert import modeling_bert
from transformers import BertConfig, BertModel, BertTokenizer
from transformers import AlbertConfig, AlbertModel, BertTokenizer
from transformers import AutoConfig, AutoModel, AutoTokenizer
from total_utils.DataExample import DataExample
from total_utils.common import get_logger
import numpy as np
import os
import datetime
import random


class Config(object):
    def __init__(self):
        self.gpu_id = 0
        self.use_multi_gpu = False
        self.pre_model = ["bert", "albert-zh", "auto"][0]

        # 基本参数
        self.random_seed = 7
        self.train_epoch = 20
        self.train_batch_size = 16
        self.dev_batch_size = 1
        self.test_batch_size = 1
        self.seq_length = 400
        self.lr_bert = 3e-5
        self.lr_task = 1e-4
        self.warmup_prop = 0.1

        self.decoder_dropout = 0.3
        self.decoder_num_head = 1
        self.pos_emb_size = 64


        # loss平方参数
        self.pow_0 = 1
        self.pow_1 = 1
        self.pow_2 = 1

        # 各个loss的权重参数
        self.w1 = 1.0
        self.w2 = 1.0
        self.w3 = 1.0

        # 预测时各个阈值
        self.threshold_0 = 0.5
        self.threshold_1 = 0.5
        self.threshold_2 = 0.5
        self.threshold_3 = 0.5
        self.threshold_4 = 0.5

        # 触发词和论元相关长度的统计
        self.TRI_LEN = 5
        self.ARG_LEN_DICT = {'collateral': 14, 'proportion': 37, 'obj-org': 34, 'number': 18, 'date': 27, 'sub-org': 35, 'target-company': 59, 'sub': 38, 'obj': 36, 'share-org': 19, 'money': 28, 'title': 8, 'sub-per': 15, 'obj-per': 18, 'share-per': 20, 'institution': 22, 'way': 8, 'amount': 19}


        # train device selection
        if self.use_multi_gpu:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.n_gpu = torch.cuda.device_count()
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            torch.cuda.set_device(self.gpu_id)
            print('current device:', torch.cuda.current_device())  # watch for current device
            n_gpu = 1
            self.n_gpu = n_gpu

        self.source_data_dir = "/Notebook/data_801/LiangZ/event_extract/ACL2021/06_code_new/datasets/FewFC/cascading_sampled/"
        self.source_dev_dir = "/Notebook/data_801/LiangZ/event_extract/ACL2021/06_code_new/datasets/FewFC/data/dev.json"
        self.source_test_dir = "/Notebook/data_801/LiangZ/event_extract/ACL2021/06_code_new/datasets/FewFC/data/test.json"
        self.config_file_path = "/Notebook/data_801/LiangZ/event_extract/ACL2021/06_code_new/config.py"
        self.model_save_path = "/Notebook/data_801/LiangZ/event_extract/ACL2021/06_code_new/model_save/"
        self.read_model_path = "/Notebook/data_801/LiangZ/event_extract/ACL2021/06_code_new/model_save/2021-11-10_19_59_35/best_model.bin"

        # 事件类型 事件类型对应id
        self.type_id, self.id_type, \
        self.args_id, self.id_args, \
        self.ty_args, self.ty_args_id, \
        self.args_s_id, self.args_e_id = DataExample().get_dict()

        self.type_num = len(self.type_id.keys())
        self.args_num = len(self.args_id.keys())

        self.MODEL_CLASSES = {'bert': (BertConfig, BertModel, BertTokenizer),
                              'albert-zh': (AlbertConfig, AlbertModel, BertTokenizer),
                              'auto': (AutoConfig, AutoModel, AutoTokenizer)}
        self.config_class, self.model_class, self.tokenizer_class = self.MODEL_CLASSES[self.pre_model]
        self.do_lower_case = True
        self.pretrain_model_path = "/Notebook/data_801/pretrain_model/00_torch_model/pytorch_bert_chinese_L-12_H-768_A-12/"
        # self.pretrain_model_path = "/Notebook/data_801/pretrain_model/00_torch_model/chinese_rbt3_pytorch/"
        self.config_plm = self.config_class.from_pretrained(self.pretrain_model_path)
        self.hidden_size = self.config_plm.hidden_size
        self.tokenizer = self.tokenizer_class.from_pretrained(self.pretrain_model_path, do_lower_case=self.do_lower_case, cache_dir="./")



    def train_init(self):
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.cuda.manual_seed_all(self.random_seed)
        os.environ['PYTHONHASHSEED'] = str(self.random_seed)
        torch.backends.cudnn.deterministic = True  # 保证每次结果一样
        torch.backends.cudnn.benchmark = False
        self.get_save_path()

    def get_save_path(self):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
        self.model_save_path = self.model_save_path + "bert_lr_bert_{}_lr_task_{}_{}".format(self.lr_bert, self.lr_task, timestamp)

        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

        # 将config文件写入文件夹中
        with open(self.model_save_path + "/config.txt", "w", encoding="utf8") as fw:
            with open(self.config_file_path, "r", encoding="utf8") as fr:
                content = fr.read()
                fw.write(content)

        self.logger = get_logger(self.model_save_path + "/log.log")
        self.logger.info('current device:{}'.format(torch.cuda.current_device()))  # watch for current device


