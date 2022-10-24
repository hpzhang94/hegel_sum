from __future__ import division
from __future__ import print_function

import argparse
import os
import random
import numpy as np
import torch
from torch import optim
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import SciPaperDataset, collect_function
from model import HGraph_Sum
from run import train, val, test

torch.cuda.empty_cache()
torch.set_printoptions(threshold=np.inf)
# Training settings
parser = argparse.ArgumentParser()

parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=20, help='Random seed.')
parser.add_argument('--hidden', type=int, default=1536, help='Number of hidden units.')
parser.add_argument("--model_save_path", type=str)
parser.add_argument("--data_path", type=str)
parser.add_argument("--data_name", type=str)
parser.add_argument("--model", type=str)
parser.add_argument("--subset_size", type=int, default=0)
parser.add_argument("--weight_path", type=str, default=None)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--use_lda", action="store_true")
parser.add_argument("--use_kw", action="store_true")
parser.add_argument("--use_ent", action="store_true")
parser.add_argument("--use_sec", action="store_true")


args = parser.parse_args()
print(args)
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
print(device)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

data_path = args.data_path
data_name = args.data_name
model_save_root_path = args.model_save_path
model_name = args.model
subset_size = args.subset_size


print("Reading Dataset")
print("Test on {} Data".format(subset_size if subset_size > 0 else "all"))
test_dataset = SciPaperDataset(data_root=data_path, dataset_name=data_name, split="test", subset=subset_size,
                                random_state=args.seed, lda=args.use_lda, keyword=args.use_lda, ner=args.use_ent, sec=args.use_sec, clst_min_num=5, clst_max_num=25)

test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collect_function)

print("Building Model")
hgs = HGraph_Sum(768, args.hidden, dropout=0, g_layer=2).cuda()

if args.weight_path is not None:
    print("loading summarization model from {}".format(args.weight_path))
    hgs.load_state_dict(torch.load(args.weight_path), strict=False)


best_r1 = 0
best_loss = 10000
train_loss = []
val_loss = []
step = 0

print("Start Testing")
if data_name == "arxiv":
    max_sen_num = 4
elif data_name == "pubmed":
    max_sen_num = 5
rouge1_score, rouge2_score, rougel_score, loss, all_gt, all_summaries = test(test_dataloader, hgs, max_sen_num)
print("Test Finished \n"
      "Test Rouge-2 Score is {} \n"
      "Test Rouge-1 Score is {}\n"
      "Test Rouge-L Score is {} \n"
      "Test Loss: {} \n"
      "Summary Example: {} \n"
      "GT is: {}".format(rouge2_score, rouge1_score, rougel_score, loss, all_summaries[0], all_gt[0]))