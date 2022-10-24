import argparse
import os
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel, LdaMulticore
from tqdm import tqdm
from gensim.test.utils import datapath
import gc


parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str)

# parser.add_argument("--topic_num", type=str)
args = parser.parse_args()
data_root = args.data_path
topic_num = 100
stop_words = stopwords.words('english') + ['!',',','.','?','-s','-ly','</s>','s', '(', ")", "@", "[", "]", "/", "_"]

def clean_sentence(sen):
    tokenizer = RegexpTokenizer(r'\w+')
    words = tokenizer.tokenize(sen)
    # print(words)
    return [w for w in words if w.lower() not in stop_words]


data_names = ["arxiv", "pubmed"]
splits = ["test", "val", "train"]


for dataset_name in data_names:
    corpus = []
    for split in splits:
        # read data
        data_path = os.path.join(data_root, "{}".format(dataset_name), "{}.npy".format(split))
        docs = np.load(data_path, allow_pickle=True)
        for doc in tqdm(docs):
            for sen in doc["article_text"]:
                corpus.append(clean_sentence(sen))
    del docs
    gc.collect()
    # print("corpus:{}".format(len(corpus)))
    dictionary = Dictionary(corpus)
    input_text = [dictionary.doc2bow(text) for text in corpus]
    # print("input_text:{}".format(len(input_text)))

    lda = LdaMulticore(input_text, num_topics=topic_num, id2word=dictionary, passes=1, workers=8)
    lda_save_path = datapath("model")
    lda.save(lda_save_path)
    # print(lda.print_topics(50, 3))
    # print(lda.get_document_topics(input_text[0]))

    doc_lda_cluster = {}
    for split in splits:
        data_path = os.path.join(data_root, "{}".format(dataset_name), "{}.npy".format(split))
        docs = np.load(data_path, allow_pickle=True)
        idx = 0
        for doc in tqdm(docs):
            doc_id = doc["article_id"]
            clst = [[] for i in range(0, topic_num)]
            for sen_id, sen in enumerate(doc["article_text"]):
                doc_topic = lda.get_document_topics(input_text[idx])
                if len(doc_topic) > 0:
                    topic_idx = doc_topic[0][0]
                    clst[topic_idx].append(sen_id)
                idx += 1
            doc_lda_cluster[doc_id] = clst
    # print(doc_lda_cluster)
    lda_clst_save_path = os.path.join(data_root, dataset_name, "lda_clst.npy")
    np.save(lda_clst_save_path, doc_lda_cluster)
    print("Saved:{}".format(dataset_name))
print("finished")





