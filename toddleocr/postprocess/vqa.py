import numpy as np
import torch

from toddleocr.utils.utility import load_vqa_bio_label_maps


class VQAReTokenLayoutLMPostProcess:
    def __init__(self, **kwargs):
        super().__init__()

    def __call__(self, preds, label=None, *args, **kwargs):
        pred_relations = preds["pred_relations"]
        if isinstance(preds["pred_relations"], torch.Tensor):
            pred_relations = pred_relations.numpy()
        pred_relations = self.decode_pred(pred_relations)
        if label is not None:
            return self._metric(pred_relations, label)
        else:
            return self._infer(pred_relations, *args, **kwargs)

    def _metric(self, pred_relations, label):
        return (pred_relations, label[-1], label[-2])

    def _infer(self, pred_relations, *args, **kwargs):
        ser_results = kwargs["ser_results"]
        entity_idx_dict_batch = kwargs["entity_idx_dict_batch"]
        results = []
        for (pred_relation, ser_result, entity_idx_dict) in zip(
            pred_relations, ser_results, entity_idx_dict_batch
        ):
            result = []
            used_tail_id = []
            for relation in pred_relation:
                if relation["tail_id"] in used_tail_id:
                    continue
                used_tail_id.append(relation["tail_id"])
                ocr_info_head = ser_result[entity_idx_dict[relation["head_id"]]]
                ocr_info_tail = ser_result[entity_idx_dict[relation["tail_id"]]]
                result.append((ocr_info_head, ocr_info_tail))
            results.append(result)
        return results

    def decode_pred(self, pred_relations):
        pred_relations_new = []
        for pred_relation in pred_relations:
            pred_relation_new = []
            pred_relation = pred_relation[1 : pred_relation[0, 0, 0] + 1]
            for relation in pred_relation:
                relation_new = dict()
                relation_new["head_id"] = relation[0, 0]
                relation_new["head"] = tuple(relation[1])
                relation_new["head_type"] = relation[2, 0]
                relation_new["tail_id"] = relation[3, 0]
                relation_new["tail"] = tuple(relation[4])
                relation_new["tail_type"] = relation[5, 0]
                relation_new["type"] = relation[6, 0]
                pred_relation_new.append(relation_new)
            pred_relations_new.append(pred_relation_new)
        return pred_relations_new


class DistillationRePostProcess(VQAReTokenLayoutLMPostProcess):
    def __init__(self, model_name=["Student"], key=None, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(model_name, list):
            model_name = [model_name]
        self.model_name = model_name
        self.key = key

    def __call__(self, preds, *args, **kwargs):
        output = dict()
        for name in self.model_name:
            pred = preds[name]
            if self.key is not None:
                pred = pred[self.key]
            output[name] = super().__call__(pred, *args, **kwargs)
        return output


class VQASerTokenLayoutLMPostProcess:
    def __init__(self, class_path, **kwargs):
        super().__init__()
        (label2id_map, self.id2label_map) = load_vqa_bio_label_maps(class_path)
        self.label2id_map_for_draw = dict()
        for key in label2id_map:
            if key.startswith("I-"):
                self.label2id_map_for_draw[key] = label2id_map["B" + key[1:]]
            else:
                self.label2id_map_for_draw[key] = label2id_map[key]
        self.id2label_map_for_show = dict()
        for key in self.label2id_map_for_draw:
            val = self.label2id_map_for_draw[key]
            if key == "O":
                self.id2label_map_for_show[val] = key
            if key.startswith("B-") or key.startswith("I-"):
                self.id2label_map_for_show[val] = key[2:]
            else:
                self.id2label_map_for_show[val] = key

    def __call__(self, preds, batch=None, *args, **kwargs):
        if isinstance(preds, tuple):
            preds = preds[0]
        if isinstance(preds, torch.Tensor):
            preds = preds.numpy()
        if batch is not None:
            return self._metric(preds, batch[5])
        else:
            return self._infer(preds, **kwargs)

    def _metric(self, preds, label):
        pred_idxs = preds.argmax(axis=2)
        decode_out_list = [[] for _ in range(pred_idxs.shape[0])]
        label_decode_out_list = [[] for _ in range(pred_idxs.shape[0])]
        for i in range(pred_idxs.shape[0]):
            for j in range(pred_idxs.shape[1]):
                if label[i, j] != -100:
                    label_decode_out_list[i].append(self.id2label_map[label[i, j]])
                    decode_out_list[i].append(self.id2label_map[pred_idxs[i, j]])
        return (decode_out_list, label_decode_out_list)

    def _infer(self, preds, segment_offset_ids, ocr_infos):
        results = []
        for (pred, segment_offset_id, ocr_info) in zip(
            preds, segment_offset_ids, ocr_infos
        ):
            pred = np.argmax(pred, axis=1)
            pred = [self.id2label_map[idx] for idx in pred]
            for idx in range(len(segment_offset_id)):
                if idx == 0:
                    start_id = 0
                else:
                    start_id = segment_offset_id[idx - 1]
                end_id = segment_offset_id[idx]
                curr_pred = pred[start_id:end_id]
                curr_pred = [self.label2id_map_for_draw[p] for p in curr_pred]
                if len(curr_pred) <= 0:
                    pred_id = 0
                else:
                    counts = np.bincount(curr_pred)
                    pred_id = np.argmax(counts)
                ocr_info[idx]["pred_id"] = int(pred_id)
                ocr_info[idx]["pred"] = self.id2label_map_for_show[int(pred_id)]
            results.append(ocr_info)
        return results


class DistillationSerPostProcess(VQASerTokenLayoutLMPostProcess):
    def __init__(self, class_path, model_name=["Student"], key=None, **kwargs):
        super().__init__(class_path, **kwargs)
        if not isinstance(model_name, list):
            model_name = [model_name]
        self.model_name = model_name
        self.key = key

    def __call__(self, preds, batch=None, *args, **kwargs):
        output = dict()
        for name in self.model_name:
            pred = preds[name]
            if self.key is not None:
                pred = pred[self.key]
            output[name] = super().__call__(pred, *args, batch=batch, **kwargs)
        return output
