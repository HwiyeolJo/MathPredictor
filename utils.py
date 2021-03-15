import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import json
import copy
from tqdm import tqdm
import random
import re

def Preprocessing(Tokenizer, FilePath, SplitMethod="ByPaperID", PrecedingSentNum=1, SucceedingSentNum=0, MaxTokenLen=512, AddNotationToVocab=False, Seed=42):
    AddTokens = []
#     if True:
    if False:
        print("\n====================")
        print("Cache Loading ... ")
        print("====================")
        try:
            FileObject = open("CachedPrecedingSentNum"+str(PrecedingSentNum)+str(SucceedingSentNum)+".json", 'r', encoding='utf-8')
            data = json.load(FileObject)
            train_src, train_tgt = data["train_src"], data["train_tgt"]
            valid_src, valid_tgt = data["valid_src"], data["valid_tgt"]
            test_src, test_tgt = data["test_src"], data["test_tgt"]
            FileObject.close()

            AddTokens = []
            FileObject = open("CachedVocabs.txt", 'r', encoding='utf-8')
            for line in FileObject:
                AddTokens.append(line.strip())
            FileObject.close()

            return (train_src, train_tgt, valid_src, valid_tgt, test_src, test_tgt), AddTokens

        except:
            pass
    
    CLS_id = Tokenizer.convert_tokens_to_ids(["[CLS]"])
    SEP_id = Tokenizer.convert_tokens_to_ids(["[SEP]"])
    PAD_id = Tokenizer.convert_tokens_to_ids(["[PAD]"])
    MASK_id = Tokenizer.convert_tokens_to_ids(["[MASK]"])
#     MASK_id = Tokenizer.mask_token
    Tokenizer.add_tokens(["$$"])
    DDollar_id = Tokenizer.convert_tokens_to_ids(["$$"])
    
    MAX_SEQ_LEN = MaxTokenLen
#     MAX_SEQ_LEN = 512
    SpTokenCnt = 2
#     Tokenizer.add_tokens("[EMPTY]")
#     EMPTY_id = Tokenizer.convert_tokens_to_ids(["[EMPTY]"])
    #####
    
    if SplitMethod == "A_Paper":
        FileObject = open(FilePath, 'r', encoding='utf-8')
        src = []; tgt = []
        data = json.load(FileObject)
        PrevTokens, NextTokens, EncTokens, PrevTokensList, NextTokensList = [], [], [], [], []
        random.seed(Seed)
        SelectedPaper = random.choice(list(set(d['paper_id'] for d in data)))
        print("SelectedPaperID", SelectedPaper)
        
        if SucceedingSentNum > 0:
            print("\n====================")
            print("Save All the Sentences")
            print("====================")
            pbar = tqdm(total = len(data))
            for cnt, d in enumerate(data):
                if d["paper_id"] == SelectedPaper:
                    Tokens = np.array(d['tokens'])
                    idx = np.where(Tokens == "SYMBOL")[0]
                    for n, i in enumerate(idx):
                        Tokens = list(Tokens)
                        Tokens[i] = d['symbols'][n]
                    EncTokens = Tokenizer.tokenize(' '.join(Tokens), add_special_tokens=False)#[:MAX_SEQ_LEN-SpTokenCnt]
                    NextTokensList.append(' '.join(EncTokens))
                pbar.update(1)
            pbar.close()
    
        MissingIndex = []
        pbar = tqdm(total = len(data))
        print("\n====================")
        print("Prepare Data")
        print("====================")
    
        for cnt, d in enumerate(data):
            if d["paper_id"] == SelectedPaper:
                Tokens = np.array(d['tokens'])
                idx = np.where(Tokens == "SYMBOL")[0]
                for n, i in enumerate(idx):
                    Tokens = list(Tokens)
                    Tokens[i] = d['symbols'][n] # Remove $ ~~~ $

                ### PrecedingSentNum
                EncTokens = Tokenizer.tokenize(' '.join(Tokens), add_special_tokens=False)#[:MAX_SEQ_LEN-SpTokenCnt]
                if len(EncTokens) == 0:
                    print("Error@"+str(cnt), d)
                    MissingIndex.append(cnt)
                    src.append('') # Add Dummies
                    tgt.append('')
                    continue
                if PrecedingSentNum >= 1:
                    PrevTokensList.append(' '.join(EncTokens))
                    PrevTokens = ' '.join(PrevTokensList[:-1]).split()
                if len(PrevTokensList) > PrecedingSentNum:
                    PrevTokensList = PrevTokensList[1:]
                if SucceedingSentNum >= 1:
                    NextTokens = ' '.join(NextTokensList[cnt+1:cnt+1+SucceedingSentNum])
                    NextTokens = NextTokens.split()

                EncTokens = np.array(EncTokens)
                EncTokens[np.where(EncTokens == "$")] = "$$" # To mark current mask
                EncTokens = EncTokens.tolist()

                ConcatTokens = PrevTokens + EncTokens + NextTokens
                ConcatTokens = ConcatTokens[max(-MAX_SEQ_LEN+len(ConcatTokens)+SpTokenCnt,0):]
                tgt.append(' '.join(["[CLS]"] + ConcatTokens + ["[SEP]"] + ["[PAD]"]*(max(0, MAX_SEQ_LEN-SpTokenCnt-len(ConcatTokens)))))
                ### Error Handling
                if len(tgt[-1].split()) != MAX_SEQ_LEN:
                    print(len(ConcatTokens), len(["[PAD]"]*(max(0, MAX_SEQ_LEN-SpTokenCnt-len(ConcatTokens)))))
                    print(tgt[-1])
                    print('tgt', len(tgt[-1].split()))
                    return

                ### Full Masking
        #         for n, i in enumerate(idx):
        #             Tokens[i] = "[MASK]"

                ### Same Num of Masking
                Tokens = list(Tokens)
        #         for n, i in enumerate(idx[::-1]):
                for n, i in enumerate(idx):
                    MaskNum = len(Tokenizer.tokenize(d['symbols'][n]))
        #             MaskSeq = ''.join(["[MASK]"]*MaskNum)
                    Tokens[i] = '$'+" [MASK] "*(MaskNum-2)+'$'
                EncTokens = Tokenizer.tokenize(' '.join(Tokens), add_special_tokens=False)#[:MAX_SEQ_LEN-SpTokenCnt]
                ConcatTokens = PrevTokens + EncTokens + NextTokens
                ConcatTokens = ConcatTokens[max(-MAX_SEQ_LEN+len(ConcatTokens)+SpTokenCnt,0):]
                src.append(' '.join(["[CLS]"] + ConcatTokens + ["[SEP]"] + ["[PAD]"]*(max(0, MAX_SEQ_LEN-SpTokenCnt-len(ConcatTokens)))))
                ### Error Handling
                if len(src[-1].split()) != MAX_SEQ_LEN:
                    print('src', len(src[-1].split()))
                    return
            pbar.update(1)
            ### For Debug
#             if cnt == 10000: break
        pbar.close()

        train_src, valid_src, test_src = [], [], []
        train_tgt, valid_tgt, test_tgt = [], [], []
        
        for i, d in enumerate(src):
            test_src.append(src[i])
            test_tgt.append(tgt[i])

        return (train_src, train_tgt, valid_src, valid_tgt, test_src, test_tgt), SelectedPaper

    if AddNotationToVocab:
        print("\n====================")
        print("AddNotationsToVocab")
        print("====================")
        FileObject = open(FilePath, 'r', encoding='utf-8')
        data = json.load(FileObject)
        pbar = tqdm(total = len(data))
        for cnt, d in enumerate(data):
            Symbols = np.array(d["symbols"])
#             print('Symbols', Symbols)
            if len(Symbols):
                for s in Symbols:
                    Notations = re.findall(r'(\\[\\a-zA-Z]+)', s, )
#                     print("Notations", Notations)
                    AddTokens += Notations
#                 if len(no) <= 10:
#                     AddTokens += [no]
            pbar.update(1)
#             if cnt == 100000: break
        pbar.close()
        FileObject.close()

        AddTokens = list(set(AddTokens))
        print("AddedTokens", AddTokens)
        
        pbar = tqdm(total = len(AddTokens))
        for t in AddTokens:
            Tokenizer.add_tokens(AddTokens)
            pbar.update(1)
        pbar.close()
        ### Heuristic
        Tokenizer.add_tokens(["SYMBOL"])
        Tokenizer.add_tokens(["SECTION"])
        Tokenizer.add_tokens(["EQUATION"])
        Tokenizer.add_tokens(["CITATION"])
        Tokenizer.add_tokens(["TABLE"])
        Tokenizer.add_tokens(["FIGURE"])
        Tokenizer.add_tokens(["$$"])
        NumLatexTokens = len(Tokenizer)
    
    #####
    FileObject = open(FilePath, 'r', encoding='utf-8')
    src = []; tgt = []
    data = json.load(FileObject)
    PrevTokens, NextTokens, EncTokens, PrevTokensList, NextTokensList = [], [], [], [], []
    
    ### Save All the Sentences for SucceedingSent
    if SucceedingSentNum > 0:
        print("\n====================")
        print("Save All the Sentences")
        print("====================")
        pbar = tqdm(total = len(data))
        for cnt, d in enumerate(data):
            Tokens = np.array(d['tokens'])
            idx = np.where(Tokens == "SYMBOL")[0]
            for n, i in enumerate(idx):
                Tokens = list(Tokens)
                Tokens[i] = d['symbols'][n]
            EncTokens = Tokenizer.tokenize(' '.join(Tokens), add_special_tokens=False)#[:MAX_SEQ_LEN-SpTokenCnt]
            NextTokensList.append(' '.join(EncTokens))
            pbar.update(1)
        pbar.close()
    
    MissingIndex = []
    pbar = tqdm(total = len(data))
    print("\n====================")
    print("Prepare Data")
    print("====================")
    
    for cnt, d in enumerate(data):    
        Tokens = np.array(d['tokens'])
        idx = np.where(Tokens == "SYMBOL")[0]
        #####
        if idx == []:
            continue
        ###
        for n, i in enumerate(idx):
            Tokens = list(Tokens)
            Tokens[i] = d['symbols'][n] # Remove $ ~~~ $
        if d["sentence_id"] == 0:
            PrevTokensList, PrevTokens = [], []
        ### PrecedingSentNum
        EncTokens = Tokenizer.tokenize(' '.join(Tokens), add_special_tokens=False)#[:MAX_SEQ_LEN-SpTokenCnt]
        if len(EncTokens) == 0:
            print("Error@"+str(cnt), d)
            MissingIndex.append(cnt)
            src.append('') # Add Dummies
            tgt.append('')
            continue
        if PrecedingSentNum >= 1:
            PrevTokensList.append(' '.join(EncTokens))
            PrevTokens = ' '.join(PrevTokensList[:-1]).split()
        if len(PrevTokensList) > PrecedingSentNum:
            PrevTokensList = PrevTokensList[1:]
        if SucceedingSentNum >= 1:
            NextTokens = ' '.join(NextTokensList[cnt+1:cnt+1+SucceedingSentNum])
            NextTokens = NextTokens.split()
            
        EncTokens = np.array(EncTokens)
        EncTokens[np.where(EncTokens == "$")] = "$$" # To mark current mask
        EncTokens = EncTokens.tolist()
        
        ConcatTokens = PrevTokens + EncTokens + NextTokens
        ConcatTokens = ConcatTokens[max(-MAX_SEQ_LEN+len(ConcatTokens)+SpTokenCnt,0):]
        tgt.append(' '.join(["[CLS]"] + ConcatTokens + ["[SEP]"] + ["[PAD]"]*(max(0, MAX_SEQ_LEN-SpTokenCnt-len(ConcatTokens)))))
        ### Error Handling
        if len(tgt[-1].split()) != MAX_SEQ_LEN:
            print(len(ConcatTokens), len(["[PAD]"]*(max(0, MAX_SEQ_LEN-SpTokenCnt-len(ConcatTokens)))))
            print(tgt[-1])
            print('tgt', len(tgt[-1].split()))
            return

        ### Full Masking
#         for n, i in enumerate(idx):
#             Tokens[i] = "[MASK]"

        ### Same Num of Masking
        Tokens = list(Tokens)
#         for n, i in enumerate(idx[::-1]):
        for n, i in enumerate(idx):
            MaskNum = len(Tokenizer.tokenize(d['symbols'][n]))
#             MaskSeq = ''.join(["[MASK]"]*MaskNum)
            Tokens[i] = '$'+" [MASK] "*(MaskNum-2)+'$'
        EncTokens = Tokenizer.tokenize(' '.join(Tokens), add_special_tokens=False)#[:MAX_SEQ_LEN-SpTokenCnt]
        ConcatTokens = PrevTokens + EncTokens + NextTokens
        ConcatTokens = ConcatTokens[max(-MAX_SEQ_LEN+len(ConcatTokens)+SpTokenCnt,0):]
        src.append(' '.join(["[CLS]"] + ConcatTokens + ["[SEP]"] + ["[PAD]"]*(max(0, MAX_SEQ_LEN-SpTokenCnt-len(ConcatTokens)))))
        ### Error Handling
        if len(src[-1].split()) != MAX_SEQ_LEN:
            print('src', len(src[-1].split()))
            return
        pbar.update(1)
        ### For Debug
#         if cnt == 10000: break
    pbar.close()
    
    if SplitMethod == "ByPaperID":
        try:
#             raise
            FileObject = open("SplitInfo_valid.txt", 'r', encoding="utf-8")
            ValidPaperID = [line.strip() for line in FileObject]
            FileObject.close()
            FileObject = open("SplitInfo_test.txt", 'r', encoding="utf-8")
            TestPaperID = [line.strip() for line in FileObject]
            FileObject.close()
        except:
            print(int(len(set(d['paper_id'] for d in data))*0.10)) # 10%
            random.seed(42)
            ValidPaperID = random.choices(list(set(d['paper_id'] for d in data)),
                                          k=int(len(set(d['paper_id'] for d in data))*0.20)) # 20%
            TestPaperID = sorted(ValidPaperID[len(ValidPaperID)//2:]) # 10% for Test
            ValidPaperID = sorted(ValidPaperID[:len(ValidPaperID)//2]) # 10% for Validation

            FileObject = open("SplitInfo_valid.txt", 'w', encoding='utf-8')
            for ids in ValidPaperID:
                FileObject.write(ids+'\n')
            FileObject.close()
            FileObject = open("SplitInfo_test.txt", 'w', encoding='utf-8')
            for ids in TestPaperID:
                FileObject.write(ids+'\n')
            FileObject.close()
            
        train_src, valid_src, test_src = [], [], []
        train_tgt, valid_tgt, test_tgt = [], [], []
        for i, d in enumerate(data):
            if i >= len(src): break
#             if i in MissingIndex: continue
            if d['paper_id'] in ValidPaperID:
                valid_src.append(src[i])
                valid_tgt.append(tgt[i])
            elif d['paper_id'] in TestPaperID:
                test_src.append(src[i])
                test_tgt.append(tgt[i])
            else:
                train_src.append(src[i])
                train_tgt.append(tgt[i])

    elif SplitMethod == "End":
        train_src = src[:int(len(src)*0.9)]
        valid_src = src[int(len(src)*0.9):int(len(src)*0.95)]
        test_src = src[int(len(src)*0.95):]
        train_tgt = tgt[:int(len(tgt)*0.9)]
        valid_tgt = tgt[int(len(tgt)*0.9):int(len(tgt)*0.95)]
        test_tgt = tgt[int(len(tgt)*0.95):]

    elif SplitMethod == "Random":
        ValidIdx = random.uniform(0,0.95)
        TestIdx = random.uniform(0,0.95)
        while ValidIdx <= TestIdx <= ValidIdx+0.05 or TestIdx <= ValidIdx <= TestIdx+0.05:
            ValidIdx = random.uniform(0,0.95)
            TestIdx = random.uniform(0,0.95)
        train_src = src[:int(len(src)*min(ValidIdx, TestIdx))]
        train_src += src[int(len(src)*min(ValidIdx, TestIdx)+0.05):int(len(src)*max(ValidIdx, TestIdx))]
        train_src += src[int(len(src)*max(ValidIdx, TestIdx)+0.05):]
        valid_src = src[int(len(src)*ValidIdx):int(len(src)*(ValidIdx+0.05))]
        test_src = src[int(len(src)*TestIdx):int(len(src)*(TestIdx+0.05))]
        train_tgt = tgt[:int(len(tgt)*min(ValidIdx, TestIdx))]
        train_tgt += tgt[int(len(tgt)*min(ValidIdx, TestIdx)+0.05):int(len(tgt)*max(ValidIdx, TestIdx))]
        train_tgt += tgt[int(len(tgt)*max(ValidIdx, TestIdx)+0.05):]
        valid_tgt = tgt[int(len(tgt)*ValidIdx):int(len(tgt)*(ValidIdx+0.05))]
        test_tgt = tgt[int(len(tgt)*TestIdx):int(len(tgt)*(TestIdx+0.05))]
    
#     FileObject = open("CachedPrecedingSentNum"+str(PrecedingSentNum)+str(SucceedingSentNum)+".json", 'w', encoding='utf-8')
#     data = {"train_src": train_src, "train_tgt": train_tgt,
#             "valid_src": valid_src, "valid_tgt": valid_tgt,
#             "test_src": test_src, "test_tgt": test_tgt,}
#     json.dump(data, FileObject)
#     FileObject.close()
    
    return (train_src, train_tgt, valid_src, valid_tgt, test_src, test_tgt), AddTokens