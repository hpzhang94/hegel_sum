from keybert import KeyBERT
import argparse
import os
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

def doc2clst(doc, kw_model, clst_num=10):
    txt = "".join(doc["article_text"])
    doc_id = doc["article_id"]
    clst = [[] for _ in range(clst_num)]

    kws = kw_model.extract_keywords(txt, keyphrase_ngram_range=(1, 1), stop_words='english', top_n=clst_num)
    # print(kws)
    for i, (kw, _) in enumerate(kws):
        for j, sen in enumerate(doc["article_text"]):
            if kw in sen:
                clst[i].append(j)
    return clst, doc_id


parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str)

args = parser.parse_args()
data_root = args.data_path

sentence_model = SentenceTransformer("all-MiniLM-L6-v2").cuda()
kw_model = KeyBERT(model=sentence_model)

kw_clsts = {}
splits = ["test", "val", "train"]
# splits = ["test"]
for dataset_name in ["arxiv", "pubmed"]:
# for dataset_name in ["arxiv"]:
    for split in splits:
        # read data
        data_path = os.path.join(data_root, "{}".format(dataset_name), "{}.npy".format(split))
        docs = np.load(data_path, allow_pickle=True)
        for doc in tqdm(docs):
            clst, doc_id = doc2clst(doc, kw_model)
            kw_clsts[doc_id] = clst

    print("Saving Results")
    kw_clst_save_path = os.path.join(data_root, dataset_name, "kw_clst.npy")
    np.save(kw_clst_save_path, kw_clsts)
print("Finished")