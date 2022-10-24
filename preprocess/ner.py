import argparse
import os
import numpy as np
from tqdm import tqdm
import scispacy
import spacy
from collections import Counter

from utils import is_number
import gc

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str)

args = parser.parse_args()
data_root = args.data_path
spacy.prefer_gpu()

splits = ["test", "val", "train"]


for dataset_name in ["pubmed", "arxiv"]:
# for dataset_name in ["arxiv"]:
    corpus = []
    for split in splits:
        # read data
        data_path = os.path.join(data_root, "{}".format(dataset_name), "{}.npy".format(split))
        docs = np.load(data_path, allow_pickle=True)
        for doc in tqdm(docs):
            for sen in doc["article_text"]:
                corpus.append(sen)
    del docs
    gc.collect()

    nlp = spacy.load('en_core_sci_lg')
    processed_docs = nlp.pipe(corpus, disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"])
    ents = []


    print("Performing NER")
    for doc in tqdm(processed_docs):
        sen_ents = [ent.text for ent in doc.ents if not is_number(ent.text)]
        ents.append(sen_ents)

    print("Do Clustering")
    ent_clsts = {}
    for split in splits:
        # read data
        data_path = os.path.join(data_root, "{}".format(dataset_name), "{}.npy".format(split))
        docs = np.load(data_path, allow_pickle=True)
        idx = 0
        for doc in tqdm(docs):
            doc_ents = []
            doc_edges = []
            doc_id = doc["article_id"]
            for sen_id, sen in enumerate(doc["article_text"]):
                doc_ents += ents[idx]
                idx += 1
            ents_counter = Counter(doc_ents)
            top_20_ent = ents_counter.most_common(20)
            # print(top_20_ent)
            ent_clst = [[] for i in range(20)]
            for ent_id, ent in enumerate(top_20_ent):
                ent = ent[0]

                for i, sen in enumerate(doc["article_text"]):
                    if ent in sen:
                        ent_clst[ent_id].append(i)

            ent_clsts[doc_id] = ent_clst
    print("Saving Results")
    ent_clst_save_path = os.path.join(data_root, dataset_name, "ent_clst.npy")
    np.save(ent_clst_save_path, ent_clsts)
    print("Finished")



    # print(ent_clsts[doc_id])
