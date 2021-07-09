#!/usr/bin/env python
# coding: utf-8

import os  
from spacy.lang.en import English
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import datetime
import argparse
import json

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.preprocessing import normalize
from sklearn.metrics import coverage_error,label_ranking_average_precision_score, precision_score


nlp = English()
tokenizer = nlp.tokenizer

class pyfred(nn.Module):

    def __init__(self, na, W, i2w, pl, nhid=512, L2loss=False):
        super(pyfred, self).__init__()

        self.nw = W.shape[0]
        self.r = W.shape[1]
        self.na = na
        self.pl = pl

        self.L2loss=L2loss

        self.i2w = i2w            

        self.W = nn.Embedding.from_pretrained(torch.from_numpy(W))

        self.A = nn.Embedding(self.na, self.r)
        self.decoder = nn.GRU(2*self.r, nhid, bidirectional=False, batch_first=True)
        # self.decoder = nn.LSTM(2*self.r, nhid, bidirectional=False, batch_first=True)

        self.drop = nn.Dropout(0.2)

        self.mapper = nn.Linear(nhid, self.nw)

        self.reducer = nn.Linear(nhid, self.r)

    def single_step(self, a, x, hidden):

        x = x.unsqueeze(1)

        a_embds = self.drop(self.A(a.long()))
        w_embds = self.drop(self.W(x.long()))

        x = torch.cat((a_embds, w_embds), 2)

        out, hid = self.decoder(x.float(), hidden.unsqueeze(0))
        # out, (hid, _) = self.decoder(x.float(), (hidden.unsqueeze(0), torch.randn(hidden.shape).unsqueeze(0)))

        dec = self.mapper(out.squeeze(1))

        return F.log_softmax(dec, dim=-1), hid.squeeze(0)

    def regularization(self, a, x, x_topic):

        if self.L2loss=='USE':
            w_embds = self.reducer(x_topic)
        elif self.L2loss=='w2vec':
            mask = (x!=0)

            w_embds = self.W(x.long()).sum(1)/mask.sum(1).view(-1,1)
        
        a_embds = self.A(a.long()).squeeze(1)

        return (w_embds - a_embds).float()


    def forward(self, a, src, trg, hidden, teacher_forcing_ratio = 0.5):

        batch_size = a.shape[0]
        trg_len = trg.shape[1]
        
        outputs = torch.zeros(batch_size, trg_len, self.nw)

        input = src[:,0]

        for t in range(0, trg_len):

            output, hidden = self.single_step(a, input, hidden)

            outputs[:,t] = output

            teacher_force = random.random() < teacher_forcing_ratio

            top1 = output.argmax(1)

            input = trg[:,t] if teacher_force else top1

        return outputs

    def translate(self, a, src, hidden, trg_len=30, generate=False, complete=0):

        with torch.no_grad():
            batch_size = a.shape[0]

            outputs = np.zeros((batch_size, trg_len))
            input = src[:,0]

            if generate:
                hidden=torch.randn(batch_size, 512)

            for t in range(0,trg_len):

                output, hidden = self.single_step(a, input, hidden)

                if t<complete:
                    input=src[:,t+1]
                else:
                    output = torch.exp(output)
                    val, argval = torch.topk(output, 5, axis=1)
                    val = F.normalize(val,p=1, dim=1)
                    input=argval[[i for i in range(batch_size)],torch.multinomial(val, 1)[:,0]]
                
                outputs[:,t] = input 

        outputs=np.vectorize(self.i2w.get)(outputs)

        for index in np.argwhere(output=="</S>"):
            output[index[0], index[1]+1:]=""

        return outputs

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-n','--name', default="pyfred_multi", type=str,
                        help='unique run id')
    parser.add_argument('-l','--L2loss', default=False, type=str,
                        help='Type of regularization (either USE, w2vec or None)')
    args = parser.parse_args()

    name=args.name
    L2loss = args.L2loss

    os.chdir('c:\\Users\\EnzoT\\Documents\\code\\pyfred')

    all_files = sorted([os.path.join("..\\LyricsGeneration\\lyricsFull", file) for file in os.listdir("..\\LyricsGeneration\\lyricsFull")])  # imagine you're one directory above test dir
    test_files = [os.path.join("..\\LyricsGeneration\\lyricsFull", file) for file in ["johnny-cash.txt"]]
    all_files = [*all_files, *test_files]
    # all_files = ["radiohead.txt","disney.txt", "adele.txt"]
    n_vers = 8
    data = []
    authors = []
    for file in all_files:
        author = file.split('\\')[-1].split(".")[0]
        authors.append(author)
        with open(file, 'r',encoding="utf-8") as fp:
            line = fp.readline()
            sentence = []   
            sentence.append(line.replace("\n"," newLine"))
            while line:
                line = fp.readline()
                sentence.append(line.replace("\n"," newLine"))
                if len(sentence) == n_vers:
                    sent = " ".join(sentence)
                    tok = ['<S>'] + [token.text.strip() for token in tokenizer(sent.lower()) if token.text.strip() != ''] + ['</S>']
                    data.append((author,sent,tok))
                    sentence = []  
            if len(sentence) != 0:
                sent = " ".join(sentence)
                tok = ['<S>'] + [token.text.strip() for token in tokenizer(sent.lower()) if token.text.strip() != ''] + ['</S>']
                data.append((author,sent,tok))

    df = pd.DataFrame(data, columns =['Author', 'Raw', 'Tokens']) 
    # test_df = df[df.Author == "johnny-cash"]
    # df = df[df.Author != "johnny-cash"]
    authors=df.Author.unique()
    aut2id = dict(zip(authors,range(len(authors))))
    df.head()
    
    from nltk.probability import FreqDist
    raw_data = list(df['Tokens'])
    flat_list = [item for sublist in raw_data for item in sublist]
    freq = FreqDist(flat_list)

    # ### Training Word2Vec and USE

    # print("USE encoding")
    # import tensorflow_hub as hub
    # module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
    # USE = hub.load(module_url)
    # print ("module %s loaded" % module_url)
    # D = np.asarray(USE(df["Raw"]),dtype=np.float32)
    np.save("use_s2_512.npy", D)
    D=np.load("use_lyrics_512_27.npy")

    from gensim.models import Word2Vec
    import numpy as np

    EMBEDDING_SIZE = 300
    w2v = Word2Vec(list(df['Tokens']), size=EMBEDDING_SIZE, window=10, min_count=1, negative=10, workers=10)
    word_map = {}
    word_map["<PAD>"] = 0
    word_vectors = [np.zeros((EMBEDDING_SIZE,))]
    for i, w in enumerate([w for w in w2v.wv.vocab]):
        word_map[w] = i+1
        word_vectors.append(w2v.wv[w])
    word_vectors = np.vstack(word_vectors)
    i2w = dict(zip([*word_map.values()],[*word_map]))
    nw = word_vectors.shape[0]
    na = len(aut2id)
    print("%d auteurs et %d mots" % (na,nw))

    def pad(a,shift = False):
        shape = len(a)
        max_s = max([len(x) for x in a])
        token = np.zeros((shape,max_s+1),dtype = int)
        mask  =  np.zeros((shape,max_s+1),dtype = int)
        for i,o in enumerate(a):
            token[i,:len(o)] = o
            mask[i,:len(o)] = 1
        if shift:
            return token[:,:-1],token[:,1:],max_s
        else:
            return token[:,:-1],max_s
            
    ang_tok,ang_tok_shift,ang_pl = pad([[word_map[w] for w in text] for text in raw_data],shift = True)

    authors_id = np.asarray([aut2id[i] for i in list(df['Author'])])
    authors_id = np.expand_dims(authors_id, 1)

    from sklearn.model_selection import train_test_split
    from torch.utils.data import TensorDataset, DataLoader

    test=True
    if test:
        i2w = json.load(open(f"results\\i2w_{name}.json"))
        i2w = {int(k):v for k,v in i2w.items()}
        word_vectors = np.zeros((len(i2w),EMBEDDING_SIZE))
        word_map = {v:k for k,v in i2w.items()}

    model = pyfred(na, word_vectors, i2w, ang_pl, L2loss=L2loss)

    checkpoint = torch.load(f'training_checkpoints\\{name}_best.pt', map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    aut_embds = np.load(f"results\\A_{name}.npy")

    with torch.no_grad():
        if L2loss=='USE':
            _, X_test, _, Y_test = train_test_split(D, authors_id[:,0], test_size=0.3, random_state=13)
            X_test = model.reducer(torch.tensor(D)).numpy()
        elif L2loss=='w2vec':
            _, X_test, _, Y_test = train_test_split(ang_tok, authors_id[:,0], test_size=0.3, random_state=13)
            X_test = model.W(torch.tensor(X_test))
            mask = (X_test!=0)
            X_test = X_test.sum(1)/mask.sum(1).numpy()

    Y_test_proba=np.zeros((len(Y_test),na))
    Y_test_proba[[i for i in range(len(Y_test))],Y_test]=1

    Y_score = normalize(X_test, axis=1) @ normalize(aut_embds, axis=1).transpose()
    Y_pred = np.argmax(Y_score, axis=1)

    ce = coverage_error(Y_test_proba, Y_score)/na
    lr = label_ranking_average_precision_score(Y_test_proba, Y_score)
    pr = precision_score(Y_test, Y_pred, average='micro')
    print(ce, lr, pr)
    with open(os.path.join("results","comparaison.txt"), 'a+', encoding='utf-8') as file:
        file.write(f"{name}:{ce} {lr} {pr}")

    batch_size=na
    vec_test=torch.tensor(USE(test_df["Raw"]).numpy())
    vec_test=torch.tensor(USE(["All you need is love, love. Love is all you need."]).numpy())

    a_test=torch.tensor([i for i in range(na)]).repeat(vec_test.shape[0]).unsqueeze(1)
    x_test=torch.tensor([word_map["<S>"] for i in range(na)]).repeat(vec_test.shape[0]).unsqueeze(1)
    vec_test=vec_test.repeat_interleave(na, dim=0)

    test_data = DataLoader(TensorDataset(a_test, vec_test, x_test), batch_size=batch_size)

    trg_len=list(test_df.Tokens.str.len())
    for batch, [a_test, vec_test, x_test] in tqdm(enumerate(test_data), total=len(test_data)):

        output=model.translate(a_test, x_test, vec_test, trg_len=trg_len, generate=True)
        
        for aut, id in aut2id.items():
            print(f"Artist {aut}")
            print(' '.join(output[id]))
            print("\n")
            with open(os.path.join("new_songs", f"{name}_{aut}.txt"), "a+") as file:
                file.write(' '.join(output[id]).replace("newline", "\n"))
                file.write("\n")

    vec=torch.tensor(USE(["All you need is love, love. Love is all you need."]).numpy())
    a=torch.tensor([i for i in range(na)]).view(na,1)
    input=torch.tensor([word_map["<S>"], word_map["love"]]*na).view(na, -1)
    vec=torch.tile(vec, (na, 1))

    output=model.translate(a, input, vec, trg_len=50, complete=1)
    for aut, id in aut2id.items():
        print(f"Artist {aut}")
        print(' '.join(output[id]))
        print("\n")