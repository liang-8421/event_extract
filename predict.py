# -*- coding: utf-8 -*-
# @Time    : 2021/11/12 22:15
# @Author  : miliang
# @FileName: predict.py
# @Software: PyCharm


from config import Config
from total_utils.EventDataLoader import EventDataLoader
from total_utils.predict_utils import test_predict


if __name__ == '__main__':
    config = Config()
    test_iter = EventDataLoader(config).get_dataloader(data_file=config.source_data_dir + "test.json", task="test")
    model_path = "/Notebook/data_801/LiangZ/event_extract/ACL2021/06_code_new/model_save/lr_bert_5e-05_lr_task_0.0001_2021-11-13_09_36_56/best_model.bin"
    test_predict(config, model_path, test_iter)