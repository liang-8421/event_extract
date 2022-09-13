# -*- coding: utf-8 -*-
# @Time    : 2021/11/12 17:00
# @Author  : miliang
# @FileName: train_fine_tune.py
# @Software: PyCharm

from config import Config
from total_utils.EventDataLoader import EventDataLoader
from total_utils.train_utils import train
from total_utils.predict_utils import test_predict


if __name__ == '__main__':
    config = Config()
    config.train_init()
    train_iter = EventDataLoader(config).get_dataloader(data_file=config.source_data_dir + "train.json", task="train")
    dev_iter = EventDataLoader(config).get_dataloader(data_file=config.source_data_dir + "dev.json", task="dev")
    train(config, train_iter, dev_iter)
    test_iter = EventDataLoader(config).get_dataloader(data_file=config.source_data_dir + "test.json", task="test")
    test_predict(config, config.model_save_path+'/best_model.bin',test_iter)




