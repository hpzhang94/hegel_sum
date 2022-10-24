import argparse
import os
import random
import numpy as np
import torch
from datasets import tqdm
from torch.utils.data import DataLoader

from dataset import SciPaperDataset
from model import BERT_Encoder
import jsonlines

torch.cuda.empty_cache()
torch.set_printoptions(threshold=np.inf)
# Training settings
parser = argparse.ArgumentParser()

parser.add_argument("--data_path", type=str)
parser.add_argument("--data_name", type=str)
parser.add_argument("--save_path", type=str)
args = parser.parse_args()
print(args)
data_path = args.data_path
data_name = args.data_name
save_root_path = args.save_path

bert_encoder = BERT_Encoder().cuda()
for p in bert_encoder.parameters():
    p.requires_grad = False
for data_name in ["pubmed", "arxiv"] :
    for split in ["val", "test", "train"]:
        dataset = SciPaperDataset(data_root=data_path, dataset_name=data_name, split=split, subset=0)
        dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=0)

        for i, data_batch in tqdm(enumerate(dataloader)):
            idx = data_batch["id"][0]
            save_path = os.path.join(save_root_path, data_name, "feature", split, "{}.npz".format(idx))
            if os.path.exists(save_path):
                continue
            sen = data_batch["sen"]
            sen = [s[0] for s in sen]

            labels = data_batch["label"]
            sec_sen_num = data_batch["sec_sen_num"].squeeze(0)
            features, edges, sen_num = bert_encoder(sen, sec_sen_num)
            data = {"features": features,
                    "edges": edges}


            np.savez_compressed(save_path, feature=features.cpu(), edge=edges.cpu())
