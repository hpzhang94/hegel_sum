from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import datasets


def train(train_dataloader, model, optimizer, step, writer=None):
    model.train()
    loss = 0
    batch_num = 0
    print_epo = 1000

    for i, data in enumerate(train_dataloader):
        if i % print_epo == 0 and i > 0:
            batch_loss,  batch_size = train_batch(data, model,  optimizer, if_print=True)
        else:
            batch_loss,  batch_size = train_batch(data, model,  optimizer, if_print=False)

        loss += batch_loss
        batch_num += 1

        if i % print_epo == 0 and i > 0:
            print("Batch {}, Loss: {}".format(i, loss / batch_num))
            sys.stdout.flush()
        if writer is not None:
            writer.add_scalar('Loss/train', loss / batch_num, step)
        step += 1
    return loss / batch_num, step

def train_batch(data_batch, model, optimizer, if_print=True):
    optimizer.zero_grad()

    labels = data_batch["label"]
    feature = data_batch["feature"]
    edge = data_batch["edge"]
    sen_num = data_batch["sen_num"]
    mask = data_batch["mask"]
    sec_pos_label = data_batch["sec_pos_label"]
    in_sec_pos_label = data_batch["insec_pos_label"]

    x = model(feature.cuda(), edge.cuda(), sen_num.cuda(), mask.cuda(), sec_pos_label=sec_pos_label.cuda(),
              in_sec_pos_label=in_sec_pos_label.cuda())


    loss = F.binary_cross_entropy_with_logits(x.squeeze(-1), labels.cuda(), weight=mask.squeeze(-1).cuda())
    loss.backward()

    optimizer.step()

    if if_print:
        print("summary_sen_idx:{}".format(
            torch.argsort(x.squeeze(-1), dim=-1, descending=True)[0, 0:15]))
        print("output:{}".format(torch.sigmoid(x[0].squeeze(-1))))
        print("labels: {}".format(np.argwhere(labels[0].detach() == 1)))

    return loss.data, x.shape[0]


def val(val_dataloader, model):
    model.eval()
    rouge = datasets.load_metric('rouge')
    loss = 0

    batch_num = 0
    rouge1_score = []
    rouge2_score = []
    rougel_score = []
    data_num = 0

    all_summaries = []
    all_gt = []
    for i, data in enumerate(val_dataloader):
        cur_loss, scores, batch_size = val_batch(data, model)

        loss += cur_loss
        data_num += batch_size

        ranked_score_idxs = get_summary_ids(scores)
        for j in range(batch_size):
            sen = data["sen"][j]
            abs_text = "".join([sen.replace("<S>", "").replace("</S>", "") for sen in data["abs_text"][j]])
            ranked_score_idx = ranked_score_idxs[j]
            summary_text = get_summary_gt_text(ranked_score_idx, sen, max_sen_num=4) # pubmed 5
            all_gt.append(abs_text)
            all_summaries.append("".join(summary_text))
            data_num += 1

        batch_num += 1
    # print(len(all_summaries))
    rouge_results = rouge.compute(predictions=all_summaries, references=all_gt, use_stemmer=True)
    rouge1_score.append(rouge_results["rouge1"].mid.fmeasure)
    rouge2_score.append(rouge_results["rouge2"].mid.fmeasure)
    rougel_score.append(rouge_results["rougeL"].mid.fmeasure)

    rouge1_score = np.mean(rouge1_score)
    loss = loss / batch_num


    return rouge1_score, loss



def val_batch(data_batch, model):
    labels = data_batch["label"]
    feature = data_batch["feature"]
    edge = data_batch["edge"]
    sen_num = data_batch["sen_num"]
    mask = data_batch["mask"]
    sec_pos_label = data_batch["sec_pos_label"]
    in_sec_pos_label = data_batch["insec_pos_label"]

    x = model(feature.cuda(), edge.cuda(), sen_num.cuda(), mask.cuda(), sec_pos_label.cuda(), in_sec_pos_label.cuda())

    loss = F.binary_cross_entropy_with_logits(x.squeeze(-1), labels.cuda(), weight=mask.squeeze(-1).cuda()) #, pos_weight=torch.tensor(20).cuda()

    scores = torch.sigmoid(x.squeeze(-1))

    return loss.data, scores, x.shape[0]



def test(test_dataloader, model, max_sen_num=4):
    model.eval()
    rouge = datasets.load_metric('rouge')
    loss = 0

    batch_num = 0
    rouge1_score = []
    rouge2_score = []
    rougel_score = []
    data_num = 0

    all_summaries = []
    all_gt = []
    for i, data in enumerate(test_dataloader):
        cur_loss, scores, batch_size = val_batch(data, model)

        loss += cur_loss
        data_num += batch_size
        ranked_score_idxs = get_summary_ids(scores)
        for j in range(batch_size):
            sen = data["sen"][j]
            abs_text = "\n".join([sen.replace("<S>", "").replace("</S>", "") for sen in data["abs_text"][j]])
            ranked_score_idx = ranked_score_idxs[j]

            summary_text = get_summary_gt_text(ranked_score_idx, sen, max_sen_num=max_sen_num)
            all_gt.append(abs_text)
            all_summaries.append("\n".join(summary_text))
            data_num += 1

        batch_num += 1
    rouge_results = rouge.compute(predictions=all_summaries, references=all_gt, use_stemmer=True)
    rouge1_score.append(rouge_results["rouge1"].mid.fmeasure)
    rouge2_score.append(rouge_results["rouge2"].mid.fmeasure)
    rougel_score.append(rouge_results["rougeLsum"].mid.fmeasure)

    rouge1_score = np.mean(rouge1_score)
    rouge2_score = np.mean(rouge2_score)
    rougel_score = np.mean(rougel_score)
    loss = loss / batch_num


    return rouge1_score, rouge2_score, rougel_score, loss, all_gt, all_summaries


def get_summary_ids(scores):
    # scores : (batch_size, sen_len)
    return torch.argsort(scores, dim=1, descending=True)


def get_summary_gt_text(ranked_score_idxs, sen,  max_sen_num=8, oracle=False, gt=None):
    # get summary
    summary_text = []
    if oracle:
        for idx in gt:
                summary_text.append(sen[idx])
    else:
        for i, idx in enumerate(ranked_score_idxs):
            if idx < len(sen):
                summary_text.append(sen[idx])
            if len(summary_text) >= max_sen_num:
                break
    return summary_text

def get_summary_gt_text_w(ranked_score_idxs, sen, max_word_num=200):
    # get summary
    summary_text = []
    summary_word_num = 0
    for idx in ranked_score_idxs:
        if idx < len(sen):
            summary_text.append(sen[idx])
            summary_word_num += len(sen[idx].split())
        if summary_word_num >= max_word_num:
            break
    return summary_text