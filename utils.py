import torch


def build_hypergraph(sec_lens, sen_num, word_nums):
    #build section hyper graph
    sec_edges = []
    start_idx = 0
    # print("sec_lens:{}".format(sec_lens))
    for sec_len in sec_lens:
        sec_edge = torch.zeros(sen_num + torch.sum(word_nums))
        sec_edge[start_idx:start_idx+sec_len] = 1
        sec_edges.append(sec_edge)
        start_idx += sec_len

    #build sentence edges
    sen_edges = []
    start_idx = sen_num
    for sen_id, n in enumerate(word_nums):
        sen_edge = torch.zeros(sen_num + torch.sum(word_nums))
        # print(sen_edge.shape)
        sen_edge[sen_id] = 1
        sen_edge[start_idx:start_idx + n] = 1
        sen_edges.append(sen_edge)
        start_idx += n

    edges = sec_edges + sen_edges
    edges = torch.stack(edges, dim=0)
    return edges


def cluster_to_hedge(cluster, node_num, threshold_low=3, threshold_high=1000000):
    hedges = []
    for c in cluster:
        if threshold_low < len(c) <= threshold_high:
            edge = torch.zeros(node_num)
            edge[c] = 1
            hedges.append(edge)
        else:
            continue
    if len(hedges) == 0:
        hedges.append(torch.zeros(node_num))

    return torch.stack(hedges, dim=0)


def section_to_clst(section):
    clst = []
    idx = 0
    for sec in section:
        c = []
        for _ in sec:
            c.append(idx)
            idx += 1
        clst.append(c)
    return clst

def get_pos_label(sec_clst):
    in_sec_pos_label = []
    sec_pos_label = []
    for sec_id, sec in enumerate(sec_clst):
        for idx, _ in enumerate(sec):
            in_sec_pos_label.append(idx + 1)
            sec_pos_label.append(sec_id + 1)
    return sec_pos_label, in_sec_pos_label


