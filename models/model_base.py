# -*- coding: utf-8 -*-
# @Time    : 2021/11/12 17:03
# @Author  : miliang
# @FileName: model_base.py
# @Software: PyCharm


from torch import nn
import torch
from models.layers import AdaptiveAdditionPredictor, ConditionalLayerNorm, MultiHeadedAttention, gelu
import numpy as np
from total_utils.EventDataLoader import get_relative_pos, get_trigger_mask



class TypeCls(nn.Module):
    def __init__(self, config):
        super(TypeCls, self).__init__()
        self.type_emb = nn.Embedding(config.type_num, config.hidden_size)
        self.register_buffer('type_indices', torch.arange(0, config.type_num, 1).long())
        self.dropout = nn.Dropout(config.decoder_dropout)

        self.Predictor = AdaptiveAdditionPredictor(config.hidden_size, dropout_rate=config.decoder_dropout)

    def forward(self, text_rep, mask):
        type_emb = self.type_emb(self.type_indices)
        pred = self.Predictor(type_emb, text_rep, mask)  # [b, c]
        p_type = torch.sigmoid(pred)
        return p_type, type_emb


class TriggerRec(nn.Module):
    def __init__(self, config):
        super(TriggerRec, self).__init__()
        self.ConditionIntegrator = ConditionalLayerNorm(config.hidden_size)
        self.SA = MultiHeadedAttention(config.hidden_size, heads_num=config.decoder_num_head,
                                       dropout=config.decoder_dropout)

        self.hidden = nn.Linear(config.hidden_size, config.hidden_size)
        self.head_cls = nn.Linear(config.hidden_size, 1, bias=True)
        self.tail_cls = nn.Linear(config.hidden_size, 1, bias=True)

        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.decoder_dropout)

    def forward(self, query_emb, text_emb, mask):
        '''

        :param query_emb: [b, e]
        :param text_emb: [b, t, e]
        :param mask: 0 if masked
        :return: [b, t, 1], [], []
        '''

        h_cln = self.ConditionIntegrator(text_emb, query_emb)

        h_cln = self.dropout(h_cln)
        h_sa = self.SA(h_cln, h_cln, h_cln, mask)
        h_sa = self.dropout(h_sa)
        inp = self.layer_norm(h_sa + h_cln)
        inp = gelu(self.hidden(inp))
        inp = self.dropout(inp)
        p_s = torch.sigmoid(self.head_cls(inp))  # [b, t, 1]
        p_e = torch.sigmoid(self.tail_cls(inp))  # [b, t, 1]
        return p_s, p_e, h_cln


class ArgsRec(nn.Module):
    def __init__(self, config):
        super(ArgsRec, self).__init__()
        self.relative_pos_embed = nn.Embedding(config.seq_length * 2, config.pos_emb_size)
        self.ConditionIntegrator = ConditionalLayerNorm(config.hidden_size)
        self.SA = MultiHeadedAttention(config.hidden_size, heads_num=config.decoder_num_head,
                                       dropout=config.decoder_dropout)
        self.hidden = nn.Linear(config.hidden_size + config.pos_emb_size, config.hidden_size)

        self.head_cls = nn.Linear(config.hidden_size, config.args_num, bias=True)
        self.tail_cls = nn.Linear(config.hidden_size, config.args_num, bias=True)

        self.gate_linear = nn.Linear(config.hidden_size, config.args_num)

        self.dropout = nn.Dropout(config.decoder_dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size)

    def forward(self, text_emb, relative_pos, trigger_mask, mask, type_emb):
        '''
        :param query_emb: [b, 4, e]
        :param text_emb: [b, t, e]
        :param relative_pos: [b, t, e]
        :param trigger_mask: [b, t]
        :param mask:
        :param type_emb: [b, e]
        :return:  [b, t, a], []
        '''
        trigger_emb = torch.bmm(trigger_mask.unsqueeze(1).float(), text_emb).squeeze(1)  # [b, e]
        trigger_emb = trigger_emb / 2

        h_cln = self.ConditionIntegrator(text_emb, trigger_emb)
        h_cln = self.dropout(h_cln)
        h_sa = self.SA(h_cln, h_cln, h_cln, mask)
        h_sa = self.dropout(h_sa)
        h_sa = self.layer_norm(h_sa + h_cln)

        rp_emb = self.relative_pos_embed(relative_pos)
        rp_emb = self.dropout(rp_emb)

        inp = torch.cat([h_sa, rp_emb], dim=-1)

        inp = gelu(self.hidden(inp))
        inp = self.dropout(inp)

        p_s = torch.sigmoid(self.head_cls(inp))  # [b, t, l]
        p_e = torch.sigmoid(self.tail_cls(inp))

        type_soft_constrain = torch.sigmoid(self.gate_linear(type_emb))  # [b, l]
        type_soft_constrain = type_soft_constrain.unsqueeze(1).expand_as(p_s)
        p_s = p_s * type_soft_constrain
        p_e = p_e * type_soft_constrain

        return p_s, p_e, type_soft_constrain


class CasEE(nn.Module):
    def __init__(self, config):
        super(CasEE, self).__init__()
        # config参数传递

        self.type_num = config.type_num
        self.args_num = config.args_num
        self.TRI_LEN = config.TRI_LEN
        self.device = config.device


        self.pow_0 = config.pow_0
        self.pow_1 = config.pow_1
        self.pow_2 = config.pow_2

        self.w1 = config.w1
        self.w2 = config.w2
        self.w3 = config.w3

        self.threshold_0 = config.threshold_0
        self.threshold_1 = config.threshold_1
        self.threshold_2 = config.threshold_2
        self.threshold_3 = config.threshold_3
        self.threshold_4 = config.threshold_4

        self.seq_length = config.seq_length
        self.id_type = config.id_type
        self.ty_args_id = config.ty_args_id
        self.id_args = config.id_args
        self.args_len_dict = config.ARG_LEN_DICT


        self.pre_model = config.model_class.from_pretrained(config.pretrain_model_path)

        self.type_cls = TypeCls(config)
        self.trigger_rec = TriggerRec(config)
        self.args_rec = ArgsRec(config)

        self.loss_0 = nn.BCELoss(reduction='none')
        self.loss_1 = nn.BCELoss(reduction='none')
        self.loss_2 = nn.BCELoss(reduction='none')

    def forward(self,  data_type_id_s, type_vec_s,
                input_ids, segment_ids, input_mask,
                r_pos, t_m, t_s, t_e, a_s, a_e, a_m):
        output_emb = self.pre_model(
            input_ids=input_ids,
            attention_mask=input_mask,
            token_type_ids=segment_ids,
        )[0]
        p_type, type_emb = self.type_cls(output_emb, input_mask)
        p_type = p_type.pow(self.pow_0)
        type_loss = self.loss_0(p_type, type_vec_s)
        type_loss = torch.sum(type_loss)

        type_rep = type_emb[data_type_id_s, :]
        p_s, p_e, text_rep_type = self.trigger_rec(type_rep, output_emb, input_mask)
        p_s = p_s.pow(self.pow_1)
        p_e = p_e.pow(self.pow_1)
        p_s = p_s.squeeze(-1)
        p_e = p_e.squeeze(-1)
        trigger_loss_s = self.loss_1(p_s, t_s)
        trigger_loss_e = self.loss_1(p_e, t_e)
        mask_t = input_mask.float()  # [b, t]
        trigger_loss_s = torch.sum(trigger_loss_s.mul(mask_t))
        trigger_loss_e = torch.sum(trigger_loss_e.mul(mask_t))

        p_s, p_e, type_soft_constrain = self.args_rec(text_rep_type, r_pos, t_m, input_mask, type_rep)
        p_s = p_s.pow(self.pow_2)
        p_e = p_e.pow(self.pow_2)
        args_loss_s = self.loss_2(p_s, a_s.transpose(1, 2))  # [b, t, l]
        args_loss_e = self.loss_2(p_e, a_e.transpose(1, 2))
        mask_a = input_mask.unsqueeze(-1).expand_as(args_loss_s).float()  # [b, t, l]
        args_loss_s = torch.sum(args_loss_s.mul(mask_a))
        args_loss_e = torch.sum(args_loss_e.mul(mask_a))

        trigger_loss = trigger_loss_s + trigger_loss_e
        args_loss = args_loss_s + args_loss_e

        type_loss = self.w1 * type_loss
        trigger_loss = self.w2 * trigger_loss
        args_loss = self.w3 * args_loss
        loss = type_loss + trigger_loss + args_loss

        return loss, type_loss, trigger_loss, args_loss

    def plm(self, input_ids, segment_ids, input_mask):
        assert input_ids.size(0) == 1
        output_emb = self.pre_model(
            input_ids=input_ids,
            attention_mask=input_mask,
            token_type_ids=segment_ids,
        )[0]
        return output_emb

    def predict_type(self, text_emb, mask):
        assert text_emb.size(0) == 1
        p_type, type_emb = self.type_cls(text_emb, mask)
        p_type = p_type.view(self.type_num).data.cpu().numpy()
        return p_type, type_emb

    def predict_trigger(self, type_rep, text_emb, mask):
        assert text_emb.size(0) == 1
        p_s, p_e, text_rep_type = self.trigger_rec(type_rep, text_emb, mask)
        p_s = p_s.squeeze(-1)  # [b, t]
        p_e = p_e.squeeze(-1)
        mask = mask.float()  # [1, t]
        p_s = p_s.mul(mask)
        p_e = p_e.mul(mask)
        p_s = p_s.view(self.seq_length).data.cpu().numpy()  # [b, t]
        p_e = p_e.view(self.seq_length).data.cpu().numpy()
        return p_s, p_e, text_rep_type

    def predict_args(self, text_rep_type, relative_pos, trigger_mask, mask, type_rep):
        assert text_rep_type.size(0) == 1
        p_s, p_e, type_soft_constrain = self.args_rec(text_rep_type, relative_pos, trigger_mask, mask, type_rep)
        mask = mask.unsqueeze(-1).expand_as(p_s).float()  # [b, t, l]
        p_s = p_s.mul(mask)
        p_e = p_e.mul(mask)
        p_s = p_s.view(self.seq_length, self.args_num).data.cpu().numpy()
        p_e = p_e.view(self.seq_length, self.args_num).data.cpu().numpy()
        return p_s, p_e, type_soft_constrain

    def predict(self, input_ids, segment_ids, input_mask, data_ids, content):
        content = content[0]
        result = {'id': data_ids[0], 'content': content}
        text_emb = self.plm(input_ids=input_ids,
                            segment_ids=segment_ids,
                            input_mask=input_mask)

        p_type, type_emb = self.predict_type(text_emb, input_mask)  # 预测事件类型
        type_pred = np.array(p_type > self.threshold_0, dtype=bool)
        type_pred = [i for i, t in enumerate(type_pred) if t]
        events_pred = []

        # 预测事件类型
        for type_pred_one in type_pred:
            type_rep = type_emb[type_pred_one, :]
            type_rep = type_rep.unsqueeze(0)
            p_s, p_e, text_rep_type = self.predict_trigger(type_rep, text_emb, input_mask)
            trigger_s = np.where(p_s > self.threshold_1)[0]
            trigger_e = np.where(p_e > self.threshold_2)[0]

            trigger_spans = []

            for i in trigger_s:
                es = trigger_e[trigger_e >= i]
                if len(es) > 0:
                    e = es[0]
                    if e - i + 1 <= self.TRI_LEN:
                        trigger_spans.append((i, e))

            for k, span in enumerate(trigger_spans):
                rp = get_relative_pos(span[0], span[1], self.seq_length)
                rp = [p + self.seq_length for p in rp]
                tm = get_trigger_mask(span[0], span[1], self.seq_length)
                rp = torch.LongTensor(rp).to(self.device)
                tm = torch.LongTensor(tm).to(self.device)
                rp = rp.unsqueeze(0)
                tm = tm.unsqueeze(0)

                p_s, p_e, type_soft_constrain = self.predict_args(text_rep_type, rp, tm, input_mask, type_rep)

                p_s = np.transpose(p_s)
                p_e = np.transpose(p_e)

                type_name = self.id_type[type_pred_one]
                pred_event_one = {'type': type_name}
                pred_trigger = {'span': [int(span[0]), int(span[1]) + 1],
                                'word': content[int(span[0]):int(span[1]) + 1]}
                pred_event_one['trigger'] = pred_trigger
                pred_args = {}
                args_candidates = self.ty_args_id[type_pred_one]

                for i in args_candidates:
                    pred_args[self.id_args[i]] = []
                    args_s = np.where(p_s[i] > self.threshold_3)[0]
                    args_e = np.where(p_e[i] > self.threshold_4)[0]
                    for j in args_s:
                        es = args_e[args_e >= j]
                        if len(es) > 0:
                            e = es[0]
                            if e - j + 1 <= self.args_len_dict[self.id_args[i]]:
                                pred_arg = {'span': [int(j), int(e) + 1], 'word': content[int(j):int(e) + 1]}
                                pred_args[self.id_args[i]].append(pred_arg)
                pred_event_one['args'] = pred_args
                events_pred.append(pred_event_one)

        result['events'] = events_pred
        return result
