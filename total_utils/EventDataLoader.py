# -*- coding: utf-8 -*-
# @Time    : 2021/11/12 14:50
# @Author  : miliang
# @FileName: dataiter.py
# @Software: PyCharm


from total_utils.DataExample import DataExample
import numpy as np
from torch.utils.data import Dataset, RandomSampler, SequentialSampler, DataLoader
import torch
from config import Config


class FeatureDataset(Dataset):
    """Pytorch Dataset for InputFeatures
    """

    def __init__(self, features):
        self.features = features

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index]


class InputFeature(object):
    def __init__(self,
                 data_ids=None,
                 content=None,
                 input_ids=None,
                 segment_ids=None,
                 input_mask=None,
                 data_type_id_s=None,
                 type_vec_s=None,
                 r_pos=None,
                 t_m=None,
                 t_s=None,
                 t_e=None,
                 a_s=None,
                 a_e=None,
                 a_m=None
                 ):
        self.data_ids = data_ids
        self.content = content
        self.input_ids = input_ids
        self.segment_ids = segment_ids
        self.input_mask = input_mask
        self.data_type_id_s = data_type_id_s
        self.type_vec_s = type_vec_s
        self.r_pos = r_pos
        self.t_m = t_m
        self.t_s = t_s
        self.t_e = t_e
        self.a_s = a_s
        self.a_e = a_e
        self.a_m = a_m


def get_relative_pos(start_idx, end_idx, length):
    '''
    return relative position
    [start_idx, end_idx]
    '''
    pos = list(range(-start_idx, 0)) + [0] * (end_idx - start_idx + 1) + list(range(1, length - end_idx))
    return pos


def get_trigger_mask(start_idx, end_idx, length):
    '''
        used to generate trigger mask, where the element of start/end postion is 1
        [000010100000]
        '''
    mask = [0] * length
    mask[start_idx] = 1
    mask[end_idx] = 1
    return mask


class EventDataLoader(object):
    def __init__(self, config):
        self.config = config
        self.device = config.device
        self.tokenizer = config.tokenizer
        self.seq_length = config.seq_length
        self.train_batch_size = config.train_batch_size
        self.dev_batch_size = config.dev_batch_size
        self.test_batch_size = config.test_batch_size

        # 事件参数
        self.type_id = config.type_id
        self.type_num = config.type_num

        self.args_num = config.args_num
        self.args_s_id = config.args_s_id
        self.args_e_id = config.args_e_id



    def data_to_id(self, data_contents):
        data_contents = [token.lower() for token in data_contents]
        inputs = self.tokenizer.encode_plus(data_contents, add_special_tokens=True, max_length=self.seq_length,
                                            truncation=True, padding='max_length')
        input_ids, segment_ids, input_mask = inputs["input_ids"], inputs["token_type_ids"], inputs['attention_mask']
        return input_ids, segment_ids, input_mask

    def type_to_id(self, data_type, data_occur):
        data_type_id = self.type_id[data_type]
        type_vec = np.array([0] * self.type_num)
        for occ in data_occur:
            idx = self.type_id[occ]
            type_vec[idx] = 1
        return data_type_id, type_vec

    def get_rp_tm(self, trigger, index):
        span = trigger[index]
        pos = get_relative_pos(span[0], span[1] - 1, self.seq_length)
        pos = [p + self.seq_length for p in pos]
        mask = get_trigger_mask(span[0], span[1] - 1, self.seq_length)
        return pos, mask

    def trigger_seq_id(self, data_trigger):
        '''
            given trigger span, return ground truth trigger matrix, for bce loss
            t_s: trigger start sequence, 1 for position 0
            t_e: trigger end sequence, 1 for position 0
        '''
        t_s = [0] * self.seq_length
        t_e = [0] * self.seq_length

        for t in data_trigger:
            t_s[t[0]] = 1
            t_e[t[1] - 1] = 1

        return t_s, t_e

    def args_seq_id(self, data_args_dict):
        '''
        given argument span, return ground truth argument matrix, for bce loss
        '''
        args_s = np.zeros(shape=[self.args_num, self.seq_length])
        args_e = np.zeros(shape=[self.args_num, self.seq_length])
        arg_mask = [0] * self.args_num

        for args_name in data_args_dict:
            s_r_i = self.args_s_id[args_name + "_s"]
            e_r_i = self.args_e_id[args_name + '_e']
            arg_mask[s_r_i] = 1
            for span in data_args_dict[args_name]:
                args_s[s_r_i][span[0]] = 1
                args_e[e_r_i][span[1] - 1] = 1
        return args_s, args_e, arg_mask

    def covert_example_to_feature(self,data_file):
        data = DataExample().read_labeled_data(data_file)
        features = []
        for index, example in enumerate(data):
            input_ids, segment_ids, input_mask = self.data_to_id(example.event_content)
            data_type_id_s, type_vec_s = self.type_to_id(example.event_type, example.event_occur)
            r_pos, t_m = self.get_rp_tm(example.event_triggers, example.trigger_index)
            t_s, t_e = self.trigger_seq_id(example.event_triggers)
            a_s, a_e, a_m = self.args_seq_id(example.event_args)
            features.append(
                InputFeature(
                    data_ids=example.event_ids,
                    content=example.event_content,
                    input_ids=input_ids,
                    segment_ids=segment_ids,
                    input_mask=input_mask,
                    data_type_id_s=data_type_id_s,
                    type_vec_s=type_vec_s,
                    r_pos=r_pos,
                    t_m=t_m,
                    t_s=t_s,
                    t_e=t_e,
                    a_s=a_s,
                    a_e=a_e,
                    a_m=a_m
                )
            )
        return features

    @staticmethod
    def collate_train_fn(features):
        """
        将InputFeatures转换为Tensor
        Args:
            features (List[InputFeatures])
        Returns:
            tensors (List[Tensors])
        """
        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.int64)
        segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.int64)
        input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.int64)
        data_type_id_s = torch.tensor([f.data_type_id_s for f in features], dtype=torch.int64)
        type_vec_s = torch.tensor([f.type_vec_s for f in features], dtype=torch.float32)
        r_pos = torch.tensor([f.r_pos for f in features], dtype=torch.int64)
        t_m = torch.tensor([f.t_m for f in features], dtype=torch.int64)
        t_s = torch.tensor([f.t_s for f in features], dtype=torch.float32)
        t_e = torch.tensor([f.t_e for f in features], dtype=torch.float32)
        a_s = torch.tensor([f.a_s for f in features], dtype=torch.float32)
        a_e = torch.tensor([f.a_e for f in features], dtype=torch.float32)
        a_m = torch.tensor([f.a_m for f in features], dtype=torch.int64)

        tensors = [input_ids, segment_ids, input_mask, data_type_id_s, type_vec_s, r_pos, t_m, t_s, t_e, a_s, a_e, a_m]

        return tensors

    @staticmethod
    def collate_dev_test_fn(features):
        """
        将InputFeatures转换为Tensor
        Args:
            features (List[InputFeatures])
        Returns:
            tensors (List[Tensors])
        """
        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.int64)
        segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.int64)
        input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.int64)
        data_ids = [f.data_ids for f in features]
        content = [f.content for f in features]
        tensors = [input_ids, segment_ids, input_mask, data_ids, content]
        return tensors


    def get_dataloader(self, data_file, task):
        if task == "train":

            try:
                self.config.logger.info("reading train dataset")
            except:
                print("reading train dataset")
            features = self.covert_example_to_feature(data_file)
            dataset = FeatureDataset(features)
            try:
                self.config.logger.info("train dataset num is {}".format(len(dataset)))
            except:
                print("train dataset num is {}".format(len(dataset)))
            datasampler = RandomSampler(dataset)
            dataloader = DataLoader(dataset,
                                    sampler=datasampler,
                                    batch_size=self.train_batch_size,
                                    collate_fn=self.collate_train_fn)
        elif task == "dev":
            try:
                self.config.logger.info("reading dev dataset")
            except:
                print("reading dev dataset")
            features = self.covert_example_to_feature(data_file)
            dataset = FeatureDataset(features)
            try:
                self.config.logger.info("dev dataset num is {}".format(len(dataset)))
            except:
                print("dev dataset num is {}".format(len(dataset)))
            datasampler = SequentialSampler(dataset)
            dataloader = DataLoader(dataset,
                                    sampler=datasampler,
                                    batch_size=self.dev_batch_size,
                                    collate_fn=self.collate_dev_test_fn)
        elif task == "test":
            try:
                self.config.logger.info("reading test dataset")
            except:
                print("reading test dataset")
            features = self.covert_example_to_feature(data_file)
            dataset = FeatureDataset(features)
            try:
                self.config.logger.info("test dataset num is {}".format(len(dataset)))
            except:
                print("test dataset num is {}".format(len(dataset)))

            datasampler = SequentialSampler(dataset)
            dataloader = DataLoader(dataset,
                                    sampler=datasampler,
                                    batch_size=self.test_batch_size,
                                    collate_fn=self.collate_dev_test_fn)
        else:
            raise ValueError("task not define !")

        return dataloader



if __name__ == '__main__':
    config = Config()
    train_iter = EventDataLoader(config).get_dataloader(data_file=config.source_data_dir + "train.json", task="train")
    print(len(train_iter))
    for i in train_iter:
        a = i[0]
        # print(i[0])


    for i in train_iter:
        b = i[0]
        # print(i[0])

    print(a)
    print("*"*100)
    print(b)