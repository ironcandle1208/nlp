import torch
import torch.nn as nn
import numpy as np

class SimpleTokenizer:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.vocab_size = 0

    def build_vocab(self, sentences):
        unique_words = set()
        for sentence in sentences:
            unique_words.update(sentence.split())
        print (f"unique_words: {unique_words}")
        
        self.word2idx = {word: idx + 1 for idx, word in enumerate(unique_words)}  # 0はPAD用
        self.word2idx['<PAD>'] = 0
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)

    def tokenize(self, sentence):
        return [self.word2idx.get(word, 0) for word in sentence.split()]

def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

sentences = [
    "apple orange banana grape"
]

tokenizer = SimpleTokenizer()
tokenizer.build_vocab(sentences)

for sentence in sentences:
    print(f"Original: {sentence}")
    print(f"Tokenized: {tokenizer.tokenize(sentence)}\n")

vocab_size = tokenizer.vocab_size
embedding_dim = 8 # 埋め込みベクトルの次元数

# Embeddingレイヤーの設定
embedding_layer = nn.Embedding(vocab_size, embedding_dim)

# サンプル分の埋め込み
for sentence in sentences:
    tokenized = tokenizer.tokenize(sentence)
    input_tensor = torch.tensor(tokenized)
    embedded = embedding_layer(input_tensor)

    print(f"Sentence: {sentence}")
    print(f"Token IDs: {tokenized}")
    print(f"Embeddings:\n{embedded}\n")


    for i in range(0,len(tokenized)-1,1):
        vec_list1 = []
        vec_list2 = []
        for j in range(0,7,1):
            vec_list1.append(embedded[i][j].item())
            vec_list2.append(embedded[i+1][j].item())

        print(f"cos_sim: {cos_sim(vec_list1,vec_list2)}")