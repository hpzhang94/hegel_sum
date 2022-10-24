import torch
from datasets import tqdm
from numpy.random import default_rng
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import os
import jsonlines
import numpy as np
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from model import HGraph_Sum
from utils import section_to_clst, cluster_to_hedge, get_pos_label


class SciPaperDataset(Dataset):
    def __init__(self, data_root, dataset_name, split, subset=0, random_state=None, lda=False, keyword=False, ner=False, sec=True, feature_source="f", clst_min_num=2, clst_max_num=20):
        super(SciPaperDataset, self).__init__()
        assert split in ["train", "val", "test"]
        self.clst_min_num = clst_min_num
        self.clst_max_num = clst_max_num
        self.data_path = os.path.join(data_root, "{}".format(dataset_name), "{}.npy".format(split))
        self.label_path = os.path.join(data_root, "{}".format(dataset_name), "label")

        self.feature_source = feature_source

        #read data
        self.docs = np.load(self.data_path, allow_pickle=True)
        print(len(self.docs))

        #encoder
        if self.feature_source == "t":
            self.model = SentenceTransformer('all-mpnet-base-v2')
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.cuda()
        elif self.feature_source == "f":
            self.feature_path = os.path.join(data_root, dataset_name, "feature")

        if sec:
            self.use_sec = True
        else:
            self.use_sec = False

        if lda:
            self.use_lda = True
            self.lda_path = os.path.join(data_root, "{}".format(dataset_name), "lda_clst.npy")
            self.lda = np.load(self.lda_path, allow_pickle=True).item()
        else:
            self.use_lda = False

        if keyword:
            self.use_kw = True
            self.kw_path = os.path.join(data_root, "{}".format(dataset_name), "kw_clst.npy")
            self.kw = np.load(self.kw_path, allow_pickle=True).item()
        else:
            self.use_kw = False

        if ner:
            self.use_ent = True
            self.ent_path = os.path.join(data_root, "{}".format(dataset_name), "ent_clst.npy")
            self.ent = np.load(self.ent_path, allow_pickle=True).item()
        else:
            self.use_ent = False


        if subset > 0:
            if random_state:
                rng = default_rng(random_state)
                idxs = rng.choice(range(0, len(self.docs)), subset, replace=False)
                self.docs = self.docs[idxs]

            else:
                self.docs = self.docs[0:subset]

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, idx):
        doc = self.docs[idx]
        abstract_text = doc["abstract_text"]

        doc_id = doc["article_id"]
        # get label
        label_path = os.path.join(self.label_path, "{}.json".format(doc_id))
        with jsonlines.open(label_path) as reader:
            label = reader.read()["labels"]

        # get sentences
        sentences = doc["article_text"]

        # get features
        if self.feature_source == "t":
            feature = self.model.encode(sentences)
        elif self.feature_source == "f":
            feature = np.load(os.path.join(self.feature_path, "{}.npz".format(doc_id)))["feature"]
        # print(feature.shape)

        edge = []
        # get edges
        sec_clst = section_to_clst(doc["sections"])
        if self.use_sec:
            sec_edge = cluster_to_hedge(sec_clst, len(sentences), 0)
            edge.append(sec_edge)

        if self.use_lda:
            lda_clst = self.lda[doc_id]
            lda_edge = cluster_to_hedge(lda_clst, len(sentences), self.clst_min_num, self.clst_max_num) #(2, 15)
            edge.append(lda_edge)

        if self.use_kw:
            kw_clst = self.kw[doc_id]
            kw_edge = cluster_to_hedge(kw_clst, len(sentences), self.clst_min_num, self.clst_max_num) #(2, 20)
            edge.append(kw_edge)

        if self.use_ent:
            ent_clst = self.ent[doc_id]
            ent_edge = cluster_to_hedge(ent_clst, len(sentences), self.clst_min_num, self.clst_max_num) #(2, 20)
            edge.append(ent_edge)

        edge = torch.cat(edge, dim=0)


        sec_sen_num = [len(s) for s in doc["sections"]]

        sec_pos_label, in_sec_pos_label = get_pos_label(sec_clst)

        ret = {
            "sen": sentences,
            "feature": torch.tensor(feature, dtype=torch.float32).squeeze(0),
            "edge": edge,
            "label": torch.tensor(label, dtype=torch.float32),
            "sec_sen_num": torch.tensor(sec_sen_num, dtype=torch.int32),
            "abs_text": abstract_text,
            "sec_pos_label": torch.tensor(sec_pos_label, dtype=torch.int32),
            "insec_pos_label": torch.tensor(in_sec_pos_label, dtype=torch.int32),
            "pos_label": torch.arange(1, len(sentences) + 1),
            "id": doc_id
        }

        return ret

def collect_function(batch):
    sen = [item["sen"] for item in batch]
    abs_text = [item["abs_text"] for item in batch]
    features = [item["feature"] for item in batch]
    edges = [item["edge"] for item in batch]
    labels = [item["label"] for item in batch]
    sec_pos_label = [item["sec_pos_label"] for item in batch]
    in_sec_pos_label = [item["insec_pos_label"] for item in batch]
    pos_label = [item["pos_label"] for item in batch]
    id = [item["id"] for item in batch]

    max_node_num = max([f.shape[0] for f in features])
    if max_node_num > 5000:
        max_node_num = 5000
    # pad features
    # print(max_node_num)
    for i in range(len(features)):
        pad_len = max_node_num - features[i].shape[0]
        if pad_len > 0:
            features[i] = F.pad(features[i], (0, 0, 0, pad_len))
        else:
            features[i] = features[i][:max_node_num]
    # pad edges
    max_edge_num = max([e.shape[0] for e in edges])
    for i in range(len(edges)):
        # print(edges[i].shape)
        pad_node_len = max_node_num - edges[i].shape[1]
        pad_edge_len = max_edge_num - edges[i].shape[0]
        if pad_node_len > 0:
            edges[i] = F.pad(edges[i], (0, pad_node_len, 0, 0))
        else:
            edges[i] = edges[i][:, :max_node_num]

        if pad_edge_len > 0:
            edges[i] = F.pad(edges[i], (0, 0, 0, pad_edge_len))
        else:
            edges[i] = edges[i][:max_edge_num, :]

    max_sen_num = max([l.shape[0] for l in labels])
    masks = []

    # pad labels
    for i in range(len(features)):
        mask = torch.zeros((max_sen_num))
        mask[:labels[i].shape[0]] = 1
        masks.append(mask)

        pad_len = max_sen_num - labels[i].shape[0]
        if pad_len > 0:
            labels[i] = F.pad(labels[i], (0, pad_len))
        else:
            labels[i] = labels[i][:max_sen_num]

    for i in range(len(features)):
        pad_len = max_sen_num - sec_pos_label[i].shape[0]
        if pad_len > 0:
            sec_pos_label[i] = F.pad(sec_pos_label[i], (0, pad_len))
            in_sec_pos_label[i] = F.pad(in_sec_pos_label[i], (0, pad_len))
            pos_label[i] = F.pad(pos_label[i], (0, pad_len))
        else:
            sec_pos_label[i] = sec_pos_label[i][:max_sen_num]
            in_sec_pos_label[i] = in_sec_pos_label[i][:max_sen_num]
            pos_label[i] = pos_label[i][:max_sen_num]

    ret = {
        "sen": sen,
        "abs_text": abs_text,
        "feature": torch.stack(features),
        "edge": torch.stack(edges),
        "label": torch.stack(labels),
        "sen_num": torch.tensor(max_sen_num),
        "mask": torch.stack(masks).unsqueeze(-1),
        "sec_pos_label": torch.stack(sec_pos_label),
        "insec_pos_label": torch.stack(in_sec_pos_label),
        "pos_label": torch.stack(pos_label),
        "id": id
    }

    return ret




