# -*- coding: utf-8 -*-
# @Time    : 2021/11/12 13:15
# @Author  : miliang
# @FileName: dataload.py
# @Software: PyCharm
import json

class InputExample(object):
    def __init__(self,
                 guid=None,
                 event_ids=None,
                 event_occur=None,
                 event_type=None,
                 event_content=None,
                 event_triggers=None,
                 trigger_index=None,
                 event_args=None
                 ):
        self.guid = guid
        self.event_ids = event_ids
        self.event_occur = event_occur
        self.event_type = event_type
        self.event_content = event_content
        self.event_triggers = event_triggers
        self.trigger_index = trigger_index
        self.event_args = event_args






class DataExample(object):
    def read_labeled_data(self, file_name):
        with open(file_name, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        data = []
        for index, line in enumerate(lines):
            line_dict = json.loads(line.strip())
            data.append(
                InputExample(
                    guid=index,
                    event_ids=line_dict["id"],
                    event_occur=line_dict["occur"],
                    event_type=line_dict["type"],
                    event_content=line_dict["content"],
                    event_triggers=line_dict["triggers"],
                    trigger_index=line_dict["index"],
                    event_args=line_dict['args']

                )
            )

        return data

    def get_dict(self):
        with open("/Notebook/data_801/LiangZ/event_extract/ACL2021/06_code_new/datasets/FewFC/cascading_sampled/ty_args.json", "r", encoding="utf-8")  as fr:
            ty_args = json.load(fr)
        with open("/Notebook/data_801/LiangZ/event_extract/ACL2021/06_code_new/datasets/FewFC/cascading_sampled/shared_args_list.json", "r", encoding="utf-8") as fr:
            args_list = json.load(fr)

        args_s_id = {}
        args_e_id = {}
        for i in range(len(args_list)):
            s = args_list[i] + '_s'
            args_s_id[s] = i
            e = args_list[i] + '_e'
            args_e_id[e] = i

        id_type = {i: item for i, item in enumerate(ty_args)}
        type_id = {item: i for i, item in enumerate(ty_args)}

        id_args = {i: item for i, item in enumerate(args_list)}
        args_id = {item: i for i, item in enumerate(args_list)}

        ty_args_id = {}
        for ty in ty_args:
            args = ty_args[ty]
            tmp = [args_id[a] for a in args]
            ty_args_id[type_id[ty]] = tmp

        return type_id, id_type, args_id, id_args, ty_args, ty_args_id, args_s_id, args_e_id






