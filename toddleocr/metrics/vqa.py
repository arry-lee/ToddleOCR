import numpy as np

__all__ = ["VQAReTokenMetric", "VQASerTokenMetric"]


class VQAReTokenMetric:
    def __init__(self, main_indicator="hmean", **kwargs):
        self.main_indicator = main_indicator
        self.reset()

    def __call__(self, preds, batch, **kwargs):
        (pred_relations, relations, entities) = preds
        self.pred_relations_list.extend(pred_relations)
        self.relations_list.extend(relations)
        self.entities_list.extend(entities)

    def get_metric(self):
        gt_relations = []
        for b in range(len(self.relations_list)):
            rel_sent = []
            relation_list = self.relations_list[b]
            entitie_list = self.entities_list[b]
            head_len = relation_list[0, 0]
            if head_len > 0:
                entitie_start_list = entitie_list[1 : entitie_list[0, 0] + 1, 0]
                entitie_end_list = entitie_list[1 : entitie_list[0, 1] + 1, 1]
                entitie_label_list = entitie_list[1 : entitie_list[0, 2] + 1, 2]
                for (head, tail) in zip(
                    relation_list[1 : head_len + 1, 0],
                    relation_list[1 : head_len + 1, 1],
                ):
                    rel = {}
                    rel["head_id"] = head
                    rel["head"] = (entitie_start_list[head], entitie_end_list[head])
                    rel["head_type"] = entitie_label_list[head]
                    rel["tail_id"] = tail
                    rel["tail"] = (entitie_start_list[tail], entitie_end_list[tail])
                    rel["tail_type"] = entitie_label_list[tail]
                    rel["type"] = 1
                    rel_sent.append(rel)
            gt_relations.append(rel_sent)
        re_metrics = self.re_score(
            self.pred_relations_list, gt_relations, mode="boundaries"
        )
        metrics = {
            "precision": re_metrics["ALL"]["p"],
            "recall": re_metrics["ALL"]["r"],
            "hmean": re_metrics["ALL"]["f1"],
        }
        self.reset()
        return metrics

    def reset(self):
        self.pred_relations_list = []
        self.relations_list = []
        self.entities_list = []

    def re_score(self, pred_relations, gt_relations, mode="strict"):
        assert mode in ["strict", "boundaries"]
        relation_types = [v for v in [0, 1] if not v == 0]
        scores = {rel: {"tp": 0, "fp": 0, "fn": 0} for rel in relation_types + ["ALL"]}
        n_sents = len(gt_relations)
        n_rels = sum([len([rel for rel in sent]) for sent in gt_relations])
        n_found = sum([len([rel for rel in sent]) for sent in pred_relations])
        for (pred_sent, gt_sent) in zip(pred_relations, gt_relations):
            for rel_type in relation_types:
                if mode == "strict":
                    pred_rels = {
                        (rel["head"], rel["head_type"], rel["tail"], rel["tail_type"])
                        for rel in pred_sent
                        if rel["type"] == rel_type
                    }
                    gt_rels = {
                        (rel["head"], rel["head_type"], rel["tail"], rel["tail_type"])
                        for rel in gt_sent
                        if rel["type"] == rel_type
                    }
                elif mode == "boundaries":
                    pred_rels = {
                        (rel["head"], rel["tail"])
                        for rel in pred_sent
                        if rel["type"] == rel_type
                    }
                    gt_rels = {
                        (rel["head"], rel["tail"])
                        for rel in gt_sent
                        if rel["type"] == rel_type
                    }
                scores[rel_type]["tp"] += len(pred_rels & gt_rels)
                scores[rel_type]["fp"] += len(pred_rels - gt_rels)
                scores[rel_type]["fn"] += len(gt_rels - pred_rels)
        for rel_type in scores.keys():
            if scores[rel_type]["tp"]:
                scores[rel_type]["p"] = scores[rel_type]["tp"] / (
                    scores[rel_type]["fp"] + scores[rel_type]["tp"]
                )
                scores[rel_type]["r"] = scores[rel_type]["tp"] / (
                    scores[rel_type]["fn"] + scores[rel_type]["tp"]
                )
            else:
                (scores[rel_type]["p"], scores[rel_type]["r"]) = (0, 0)
            if not scores[rel_type]["p"] + scores[rel_type]["r"] == 0:
                scores[rel_type]["f1"] = (
                    2
                    * scores[rel_type]["p"]
                    * scores[rel_type]["r"]
                    / (scores[rel_type]["p"] + scores[rel_type]["r"])
                )
            else:
                scores[rel_type]["f1"] = 0
        tp = sum([scores[rel_type]["tp"] for rel_type in relation_types])
        fp = sum([scores[rel_type]["fp"] for rel_type in relation_types])
        fn = sum([scores[rel_type]["fn"] for rel_type in relation_types])
        if tp:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2 * precision * recall / (precision + recall)
        else:
            (precision, recall, f1) = (0, 0, 0)
        scores["ALL"]["p"] = precision
        scores["ALL"]["r"] = recall
        scores["ALL"]["f1"] = f1
        scores["ALL"]["tp"] = tp
        scores["ALL"]["fp"] = fp
        scores["ALL"]["fn"] = fn
        scores["ALL"]["Macro_f1"] = np.mean(
            [scores[ent_type]["f1"] for ent_type in relation_types]
        )
        scores["ALL"]["Macro_p"] = np.mean(
            [scores[ent_type]["p"] for ent_type in relation_types]
        )
        scores["ALL"]["Macro_r"] = np.mean(
            [scores[ent_type]["r"] for ent_type in relation_types]
        )
        return scores


class VQASerTokenMetric:
    def __init__(self, main_indicator="hmean", **kwargs):
        self.main_indicator = main_indicator
        self.reset()

    def __call__(self, preds, batch, **kwargs):
        preds, labels = preds
        self.pred_list.extend(preds)
        self.gt_list.extend(labels)

    def get_metric(self):
        from seqeval.metrics import f1_score, precision_score, recall_score

        metrics = {
            "precision": precision_score(self.gt_list, self.pred_list),
            "recall": recall_score(self.gt_list, self.pred_list),
            "hmean": f1_score(self.gt_list, self.pred_list),
        }
        self.reset()
        return metrics

    def reset(self):
        self.pred_list = []
        self.gt_list = []
