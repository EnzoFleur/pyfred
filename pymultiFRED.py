#!/usr/bin/env python
# coding: utf-8                      
import os
from spacy.lang.en import English
import pandas as pd
import numpy as np
import random
from datetime import datetime
from time import time
import argparse

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

import idr_torch 

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

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

        self.drop = nn.Dropout(0.2)

        self.mapper = nn.Linear(nhid, self.nw)

        self.reducer = nn.Linear(nhid, self.r)

    def single_step(self, a, x, hidden):

        x = x.unsqueeze(1)

        a_embds = self.drop(self.A(a.long()))
        w_embds = self.drop(self.W(x.long()))

        x = torch.cat((a_embds, w_embds), 2)

        out, hid = self.decoder(x.float(), hidden.unsqueeze(0))
        
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

            output, hidden = self.single_step(a, input, hidden.contiguous())

            outputs[:,0] = output

            teacher_force = random.random() < teacher_forcing_ratio

            top1 = output.argmax(1)

            input = trg[:,t] if teacher_force else top1

        return outputs

def train_gpus(model, train_data, optimizer, criterion, regularization, alba = None):

    model.train()

    train_loss = 0
    train_accuracy = 0
    total_step = len(train_data)
    
    for i, (x, y) in enumerate(train_data):
        if idr_torch.rank==0: start_dataload = time()

        x = x.to(gpu, non_blocking=True)
        y = y.to(gpu, non_blocking=True)

        if idr_torch.rank==0: stop_dataload = time()

        if idr_torch.rank==0: start_training = time()

        a,x_topic,x = torch.split(x,[1,512,ang_pl],dim=1)

        output = model(a, x, y, x_topic)

        output = output.view(-1, nw)

        y = y.long().view(-1)

        loss = criterion(output, y)

        if model.L2loss:
            loss += alba*regularization(model.regularization(a,x, x_topic), torch.zeros(a.shape[0], model.r))

        train_accuracy += (output.argmax(1)[y!=0] == y[y!=0]).float().mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        train_loss += loss.item()

        if idr_torch.rank==0: stop_training = time()
        if ((i + 1) % total_step//2 == 0) and (idr_torch.rank == 0):
            print('\tStep [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}, Time data load: {:.3f}ms, Time training: {:.3f}ms'.format(
                                                                    i + 1, total_step, train_loss/len(train_data), train_accuracy/len(train_data),
                                                                    (stop_dataload - start_dataload)*1000, (stop_training - start_training)*1000))

    return train_loss/len(train_data), train_accuracy/len(train_data)

def evaluate_gpus(model, test_data, criterion, regularization, alba=None):
    model.eval()

    test_loss = 0
    test_norm = 0
    test_accuracy = 0

    with torch.no_grad():

        for x, y in test_data:

            x = x.to_gpu(gpu, non_blocking=True)
            y = y.to_gpu(gpu, non_blocking=True)

            a,x_topic,x = torch.split(x,[1,512,ang_pl],dim=1)

            output = model(a, x, y, x_topic)

            output = output.view(-1, nw)

            y = y.long().view(-1)

            loss = criterion(output, y)

            if model.L2loss:
                test_norm += regularization(model.regularization(a,x, x_topic), torch.zeros(a.shape[0], model.r))

            test_accuracy += (output.argmax(1)[y!=0] == y[y!=0]).float().mean()

            test_loss += loss.item()

    return test_loss/len(test_data), test_accuracy/len(test_data), alba*test_norm/len(test_data)


if __name__ == "__main__":

    # get distributed configuration from Slurm environment
    NODE_ID = os.environ['SLURM_NODEID']
    MASTER_ADDR = os.environ['MASTER_ADDR']

    dist.init_process_group(backend='nccl', 
                        init_method='env://', 
                        world_size=idr_torch.size, 
                        rank=idr_torch.rank)

    nlp = English()
    tokenizer = nlp.tokenizer

    # display info
    if idr_torch.rank == 0:
        print(">>> Training on ", len(idr_torch.hostnames), " nodes and ", idr_torch.size, " processes, master node is ", MASTER_ADDR)
    print("- Process {} corresponds to GPU {} of node {}".format(idr_torch.rank, idr_torch.local_rank, NODE_ID))


    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch-size', default=32, type =int,
                        help='batch size. it will be divided in mini-batch for each worker')
    parser.add_argument('-e','--epochs', default=10, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-n','--name', default="pyfred_multi", type=str,
                        help='unique run id')
    parser.add_argument('-a','--alba', default=None, type=float,
                        help='Regularization coefficient')
    parser.add_argument('-l','--L2loss', default=False, type=str,
                        help='Type of regularization (either USE, w2vec or None)')
    args = parser.parse_args()

    #all_files = os.listdir("lyrics/")   # imagine you're one directory above test dir
    all_files = ["radiohead.txt","disney.txt", "adele.txt"]
    n_vers = 8
    data = []
    authors = []
    for file in all_files:
        author = file.split(".")[0]
        authors.append(author)
        with open('../../datasets/lyrics/'+file, 'r',encoding="utf-8") as fp:
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
    D=np.load("use_lyrics_512_3.npy")

    from gensim.models import Word2Vec
    import numpy as np

    EMBEDDING_SIZE = 120
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

    batch_size_per_gpu = args.batch_size
    epochs=args.epochs
    name=args.name

    batch_size = batch_size_per_gpu * idr_torch.size

    X = np.hstack([authors_id,D,ang_tok])
    Y = np.hstack([ang_tok_shift])

    X = X.astype(np.float32)

    from sklearn.model_selection import train_test_split
    from torch.utils.data import TensorDataset, DataLoader

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.80, random_state=101)

    train_data = TensorDataset(torch.tensor(X_train), torch.tensor(Y_train))
    test_data = TensorDataset(torch.tensor(X_test), torch.tensor(Y_test))

    if idr_torch.rank==0: print("Dataset is ready to be loaded !")

    torch.cuda.set_device(idr_torch.local_rank)
    gpu = torch.device("cuda")
    model = pyfred(na, word_vectors, i2w, ang_pl, L2loss=args.L2loss).to(gpu)
    ddp_model = DDP(model, device_ids=[idr_torch.local_rank])

    criterion = nn.NLLLoss(ignore_index = 0)

    if idr_torch.rank==0: print("Model is ready for training !")

    if model.L2loss:
        alba = args.alba
        if alba is None:
            print("Alba is required !")
            exit()
        regularization = nn.MSELoss()

    optimizer = optim.Adam(ddp_model.parameters(), lr=0.001)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data, 
                                                                    num_replicas=idr_torch.size,
                                                                    rank=idr_torch.rank)

    train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                                batch_size = batch_size_per_gpu,
                                                shuffle=False,
                                                num_workers=0,
                                                pin_memory=True,
                                                sampler=train_sampler)

    test_sampler = torch.utils.data.distributed.DistributedSampler(test_data, 
                                                                    num_replicas=idr_torch.size,
                                                                    rank=idr_torch.rank)

    test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                                batch_size = batch_size_per_gpu,
                                                shuffle=False,
                                                num_workers=0,
                                                pin_memory=True,
                                                sampler=test_sampler)

    best_valid_loss = float('inf')

    if idr_torch.rank==0: print(f"Training is beginning for {epochs} epochs !")
    if idr_torch.rank == 0: start = datetime.now()
    for epoch in range(1, epochs+1):

        if idr_torch.rank == 0: print(f'Epoch [{epoch}/{epochs} :')

        train_loss, train_accuracy = train_gpus(model, train_loader, optimizer, criterion, regularization, alba)
        test_loss, test_accuracy, test_L2loss = evaluate_gpus(model, test_loader, criterion, regularization, alba)

        if idr_torch.rank ==0:
            if test_loss < best_valid_loss:
                best_valid_loss = test_loss
                torch.save({'epoch':epoch,
                            'model_state_dict':model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()},  f'training_checkpoints/pyfred_{name}_{epoch}.pt')

            with open(f"results/loss_pyfred_{name}.txt", "a") as ff:
                ff.write('%06f | %06f | %06f | %06f | %06f\n' % (train_loss, test_loss, train_accuracy*100, test_accuracy*100, test_L2loss))

    if idr_torch.rank == 0:
        print(' -- Trained in ' + str(datetime.datetime.now()-start) + ' -- ')
        A = []
        with torch.no_grad():
            for i in range(model.na):
                A.append(model.A(torch.tensor(i)).numpy())
            A = np.vstack(A)
            
        np.save(f"results/author_embeddings_{name}.npy", A)