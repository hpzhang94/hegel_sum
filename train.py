from __future__ import division
from __future__ import print_function

import argparse
import os
import random
import numpy as np
import torch
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import SciPaperDataset, collect_function
from model import HGraph_Sum
from run import train, val

torch.cuda.empty_cache()
torch.set_printoptions(threshold=np.inf)
# Training settings
parser = argparse.ArgumentParser()

parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=20, help='Random seed.')
parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=5e-5, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-3, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=1536, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate (1 - keep probability).')
parser.add_argument("--model_save_path", type=str)
parser.add_argument("--data_path", type=str)
parser.add_argument("--data_name", type=str)
parser.add_argument("--model", type=str)
parser.add_argument("--subset_size", type=int, default=0)
parser.add_argument("--val_size", type=int, default=0)
parser.add_argument("--weight_path", type=str, default=None)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--use_lda", action="store_true")
parser.add_argument("--use_kw", action="store_true")
parser.add_argument("--use_ent", action="store_true")
parser.add_argument("--use_sec", action="store_true")


writer = SummaryWriter()
args = parser.parse_args()
print(args)
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
print(device)

epochs = args.epochs
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
print("Train on {} Data".format(subset_size if subset_size > 0 else "all"))
train_dataset = SciPaperDataset(data_root=data_path, dataset_name=data_name, split="train", subset=subset_size,
                                random_state=args.seed, lda=args.use_lda, keyword=args.use_lda, ner=args.use_ent,
                                sec=args.use_sec, clst_min_num=5, clst_max_num=25)
val_dataset = SciPaperDataset(data_root=data_path, dataset_name=data_name, split="val", subset=args.val_size , random_state=0,
                              lda=args.use_lda, keyword=args.use_lda, ner=args.use_ent, sec=args.use_sec, clst_min_num=5, clst_max_num=25)

train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collect_function)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collect_function)

print("Building Model")
hgs = HGraph_Sum(768, args.hidden, dropout=args.dropout, g_layer=2).cuda()

if args.weight_path is not None:
    print("loading summarization model from {}".format(args.weight_path))
    hgs.load_state_dict(torch.load(args.weight_path), strict=False)

print("Building Optimizer")
optimizer = optim.Adam(hgs.parameters(), lr=args.lr)
scheduler = ReduceLROnPlateau(optimizer, 'max', patience=6, factor=0.3, verbose=True)

best_r1 = 0
best_loss = 10000
train_loss = []
val_loss = []
step = 0

print("Start Training Summary Model")
for i in range(epochs):
    loss, step = train(train_dataloader, hgs, optimizer, step, writer)
    train_loss.append(loss)
    print("At Epoch {}, Train Loss: {}".format(i, loss))

    torch.cuda.empty_cache()

    print("Validating")
    rouge1_score, loss = val(val_dataloader, hgs)
    scheduler.step(rouge1_score)
    writer.add_scalar('Loss/val', loss, i)
    writer.add_scalar('Rouge 1/val', rouge1_score, i)
    torch.cuda.empty_cache()

    print("At Epoch {}, Val Loss: {}, Val R1: {}".format(i, loss, rouge1_score))

    if rouge1_score > best_r1:
        model_save_path = os.path.join(model_save_root_path, "{}_{}.mdl".format(i, rouge1_score))
        torch.save(hgs.state_dict(), model_save_path)

        best_r1 = rouge1_score
        print("Epoch {} Has best R1 Score of {}, saved Model to {}".format(i, best_r1, model_save_path))
print("Finished")