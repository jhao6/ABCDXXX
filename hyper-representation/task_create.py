import os
import torch
from torch.utils.data import Dataset
import numpy as np
import collections
import random
import json, pickle
from torch.utils.data import TensorDataset
GLOVE_DIM = 300
VOCAB_NAME = "glove.840B.300d.txt"
class MetaTask(Dataset):

    def __init__(self, root, split, num_task, k_support, k_query):
        """
        :param samples: list of samples
        :param num_task: number of training tasks.
        :param k_support: number of support sample per task
        :param k_query: number of query sample per task
        """
        # self.examples = examples
        # random.shuffle(self.examples)
        self.num_task = num_task
        self.k_support = k_support
        self.k_query = k_query
        self.max_seq_length = 256

        """ Read and store data from files. """
        self.labels = ["entailment", "neutral", "contradiction"]
        labels_to_idx = {label: i for i, label in enumerate(self.labels)}

        # If vocab exists on file, load it. Otherwise, read sentence data for all splits
        # from files to build vocab.
        assert split in ["train", "dev", "test"]
        self.root = os.path.join(root, "snli_1.0")
        self.split = split
        self.embed_dim = GLOVE_DIM
        self.n_classes = 3

        """ Read and store data from files. """
        self.labels = ["entailment", "neutral", "contradiction"]
        self.labels_to_idx = {label: i for i, label in enumerate(self.labels)}

        # Read sentence and label data for current split from files.
        s1_path = os.path.join(self.root, "SNLI", f"s1.{self.split}")
        s2_path = os.path.join(self.root, "SNLI", f"s2.{self.split}")
        target_path = os.path.join(self.root, "SNLI", f"labels.{self.split}")
        self.s1_sentences = [line.rstrip() for line in open(s1_path, "r")]
        self.s2_sentences = [line.rstrip() for line in open(s2_path, "r")]
        self.targets = np.array(
            [labels_to_idx[line.rstrip("\n")] for line in open(target_path, "r")]
        )
        assert len(self.s1_sentences) == len(self.s2_sentences)
        assert len(self.s1_sentences) == len(self.targets)
        self.dataset_size = len(self.s1_sentences)
        print(f"Loaded {self.dataset_size} sentence pairs for {self.split} split.")

        # If vocab exists on file, load it. Otherwise, read sentence data for all splits
        # from files to build vocab.
        vocab_path = '../data/snli_1.0/SNLI/vocab.pkl'
        if os.path.isfile(vocab_path):
            print("Loading vocab.")
            with open(vocab_path, "rb") as vocab_file:
                vocab = pickle.load(vocab_file)
        else:
            print(
                "Constructing vocab. This only needs to be done once but will take "
                "several minutes."
            )
            vocab = ["<s>", "</s>"]
            for split in ["train", "dev", "test"]:
                paths = [
                    os.path.join(self.root, "SNLI", f"s1.{split}"),
                    os.path.join(self.root, "SNLI", f"s2.{split}"),
                ]
                for path in paths:
                    for line in open(path, "r"):
                        for word in line.rstrip().split():
                            if word not in vocab:
                                vocab.append(word)
            with open(vocab_path, "wb") as vocab_file:
                pickle.dump(vocab, vocab_file)
        print(f"Loaded vocab with {len(vocab)} words.")

        # Read in GLOVE vectors and store mapping from words to vectors.
        self.word2vec = {}
        glove_path = "../data/snli_1.0/GloVe/glove.840B.300d.txt"
        wordvec_path = "../data/snli_1.0/SNLI/wordvec.pkl"
        if os.path.isfile(wordvec_path):
            print("Loading word vector mapping.")
            with open(wordvec_path, "rb") as wordvec_file:
                self.word2vec = pickle.load(wordvec_file)
        else:
            print(
                "Constructing mapping from vocab to word vectors. This only needs to "
                "be done once but can take up to 30 minutes."
            )
            with open(glove_path, "r") as glove_file:
                for line in glove_file:
                    word, vec = line.split(' ', 1)
                    if word in vocab:
                        self.word2vec[word] = np.array(list(map(float, vec.split())))
            with open(wordvec_path, "wb") as wordvec_file:
                pickle.dump(self.word_vec, wordvec_file)
        print(f"Found {len(self.word2vec)}/{len(vocab)} words with glove vectors.")

        # Split each sentence into words, add start/stop tokens to the beginning/end of
        # each sentence, and remove any words which do not have glove embeddings.
        assert "<s>" in vocab
        assert "</s>" in vocab
        assert "<s>" in self.word2vec
        assert "</s>" in self.word2vec
        for i in range(len(self.s1_sentences)):
            sent = self.s1_sentences[i]
            self.s1_sentences[i] = np.array(
                ["<s>"] +
                [word for word in sent.split() if word in self.word2vec] +
                ["</s>"]
            )
        for i in range(len(self.s2_sentences)):
            sent = self.s2_sentences[i]
            self.s2_sentences[i] = np.array(
                ["<s>"] +
                [word for word in sent.split() if word in self.word2vec] +
                ["</s>"]
            )
        self.create_batch(self.num_task)

    def create_batch(self, num_task):
        self.supports = []  # support set
        self.queries = []  # query set

        for b in range(num_task):  # for each task
            # 1.select domain randomly
            # domain = random.choice(self.examples)['domain']
            # d_examples = [e for e in self.examples if e['domain'] == domain]

            # examples = ((self.s1_sentences, self.s2_sentences), self.targets)
            label_indx = {}
            inds = []
            selected_examples = []
            selected_examples_labels =[]
            selected_examples_train_s1 = []
            selected_examples_train_s2 = []
            selected_examples_test_s1 = []
            selected_examples_test_s2 = []
            selected_examples_test= []
            exam_train_label = []
            exam_test_label = []
            selected_label = random.sample([0, 1, 2], 2)
            for i in range(len(selected_label)):
                label_indx[i] = [k for k, l in enumerate(self.targets) if l==selected_label[i]]
                selected_indx = random.sample(label_indx[i], int((self.k_support + self.k_query)/len(selected_label)))
                selected_examples_train_s1.append(np.array(self.s1_sentences)[selected_indx[:int(self.k_support/len(selected_label))]])
                selected_examples_train_s2.append(np.array(self.s2_sentences)[selected_indx[:int(self.k_support/len(selected_label))]])
                selected_examples_test_s1.append(np.array(self.s1_sentences)[selected_indx[int(self.k_support/len(selected_label)):]])
                selected_examples_test_s2.append(np.array(self.s2_sentences)[selected_indx[int(self.k_support/len(selected_label)):]])
                exam_train_label.append(np.array(self.targets)[selected_indx[:int(self.k_support/len(selected_label))]])
                exam_test_label.append(np.array(self.targets)[selected_indx[int(self.k_support/len(selected_label)):]])
            # 1.select k_support + k_query examples from domain randomly
            # selected_examples = np.array(domainExamples[0])[inds.numpy()]
            # selected_examples_labels = np.array(domainExamples[1])[inds.numpy()]
            # random.shuffle(selected_examples)
            examples_train_vec = self.word_vec(np.hstack(selected_examples_train_s1), np.hstack(selected_examples_train_s2))
            # examples_train_vec_s2 = self.word_vec(np.hstack(selected_examples_train_s2))
            examples_test_vec = self.word_vec(np.hstack(selected_examples_test_s1), np.hstack(selected_examples_test_s2))
            # examples_test_vec_s2 = self.word_vec(np.hstack(selected_examples_test_s2))


            self.supports.append((examples_train_vec, np.hstack(exam_train_label)))
            self.queries.append((examples_test_vec, np.hstack(exam_test_label)))

    # def split_words(self):
    #     # self.labels_to_idx ={'entailment':0, 'neutral':1, "contradiction":2}
    #     sentences_1 = []
    #     sentences_2 = []
    #     targets =[]
    #     for i in range(len(self.targets)):
    #         targets.append(self.targets[i])
    #         sentences_1.append(np.array(
    #             [word for word in self.s1_sentences[i].split() if word in self.word2vec]
    #         ))
    #         sentences_2.append(np.array(
    #             [word for word in self.s2_sentences[i].split() if word in self.word2vec]
    #         ))
    #     return (sentences_1, sentences_2), targets
    def word_vec(self,examples1, examples2):
        s1_embed = [None for _ in range(len(examples1))]
        s2_embed = [None for _ in range(len(examples1))]
        for i in range(len(examples1)):
            s1_embed[i] = np.zeros((len(examples1[i]), GLOVE_DIM))
            s2_embed[i] = np.zeros((len(examples2[i]), GLOVE_DIM))
            for j in range(len(examples1[i])):
                try:
                    s1_embed[i][j]=self.word2vec[examples1[i][j]]
                    s2_embed[i][j]=self.word2vec[examples2[i][j]]
                except:
                    pass
        # s1_embed = torch.from_numpy(s1_embed).float()
        return (s1_embed, s2_embed)

    def __getitem__(self, index):
        support_set = self.supports[index]
        query_set   = self.queries[index]
        return support_set, query_set


if __name__ == "__main__":
    # split, num_task, k_support, k_query, word2vec
    train_data = MetaTask("../data", "train", 500, 20, 20)
    test_data = MetaTask("../data", "test", 100, 20, 20)
    print('processed the data')
    # torch.save(train_data, '../data/snli_1.0/SNLI/train_Data.pkl')
    # torch.save(test_data, '../data/snli_1.0/SNLI/test_Data.pkl')
    print('save the train and test data')

