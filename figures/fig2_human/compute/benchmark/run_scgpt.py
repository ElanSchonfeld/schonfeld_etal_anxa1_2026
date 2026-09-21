#!/usr/bin/env python3
"""Embed mouse and Kamath cells with the pretrained scGPT human model, score KNN and logistic regression family calls, and write scgpt.csv.

Run: python run_scgpt.py
"""
import benchmark_common as bc

import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from scgpt.model import TransformerModel
from scgpt.tokenizer import GeneVocab
from scgpt.utils import load_pretrained
from scgpt.data_collator import DataCollator

if "SCGPT_MODEL_DIR" not in os.environ:
    raise RuntimeError("set SCGPT_MODEL_DIR to the scGPT_human checkpoint directory")

MODEL_DIR = Path(os.environ["SCGPT_MODEL_DIR"])
ORTHOLOGS = bc.SOURCE_ROOT / "Data" / "Mouse_Human_Gene_Orthologs.tsv"

torch.manual_seed(bc.SEED)
np.random.seed(bc.SEED)

configs = json.loads((MODEL_DIR / "args.json").read_text())
vocab = GeneVocab.from_file(MODEL_DIR / "vocab.json")
for token in ["<pad>", "<cls>", "<eoc>"]:
    if token not in vocab:
        vocab.append_token(token)

model = TransformerModel(
    ntoken=len(vocab),
    d_model=configs["embsize"],
    nhead=configs["nheads"],
    d_hid=configs["d_hid"],
    nlayers=configs["nlayers"],
    nlayers_cls=configs["n_layers_cls"],
    n_cls=1,
    vocab=vocab,
    dropout=configs["dropout"],
    pad_token=configs["pad_token"],
    pad_value=configs["pad_value"],
    do_mvc=True,
    do_dab=False,
    use_batch_labels=False,
    domain_spec_batchnorm=False,
    explicit_zero_prob=False,
    use_fast_transformer=False,
    fast_transformer_backend="flash",
    pre_norm=False,
)
device = torch.device("cpu")
load_pretrained(model, torch.load(MODEL_DIR / "best_model.pt", map_location="cpu"), verbose=False)
model.to(device)
model.eval()
model.transformer_encoder.enable_nested_tensor = False

ad_mouse = sc.read_h5ad(bc.MOUSE_H5AD)
ad_human = sc.read_h5ad(bc.KAMATH_H5AD)
y_mouse = bc.mouse_families(ad_mouse)
y_human = bc.human_families(ad_human)
train = np.isin(y_mouse, bc.FAMILIES)
test = np.isin(y_human, bc.EVAL_FAMILIES)

ortho = pd.read_csv(ORTHOLOGS, sep="\t")
mouse2human = {}
for m, h in zip(ortho["Mouse Gene"].astype(str).str.strip(), ortho["Human Gene"].astype(str).str.strip()):
    if m and h and h != "nan" and m not in mouse2human:
        mouse2human[m] = h

ortho_names, keep_idx, seen = [], [], set()
for i, g in enumerate(ad_mouse.var_names.tolist()):
    hg = mouse2human.get(g)
    if hg and hg not in seen:
        ortho_names.append(hg)
        keep_idx.append(i)
        seen.add(hg)


def vocab_subset(X, gene_names):
    ids = np.array([vocab[g] if g in vocab else -1 for g in gene_names])
    return sp.csr_matrix(X[:, ids >= 0]), ids[ids >= 0]


X_mouse, ids_mouse = vocab_subset(ad_mouse.X[:, keep_idx], ortho_names)
X_human, ids_human = vocab_subset(ad_human.X, ad_human.var_names.tolist())
del ad_mouse, ad_human

pad_token_id = vocab[configs["pad_token"]]
pad_value = configs["pad_value"]


class CellDataset(Dataset):
    def __init__(self, X, gene_ids):
        self.X = X
        self.gene_ids = gene_ids

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        row = np.asarray(self.X[idx].toarray()).ravel()
        nz = np.nonzero(row)[0]
        values = np.insert(row[nz].astype(np.float32), 0, pad_value)
        genes = np.insert(self.gene_ids[nz], 0, vocab["<cls>"])
        return {"id": idx, "genes": torch.from_numpy(genes).long(),
                "expressions": torch.from_numpy(values).float()}


def embed(X, gene_ids):
    dataset = CellDataset(X, gene_ids)
    collator = DataCollator(do_padding=True, pad_token_id=pad_token_id, pad_value=pad_value,
                            do_mlm=False, do_binning=True, max_length=1200, sampling=True,
                            keep_first_n_tokens=1)
    loader = DataLoader(dataset, batch_size=64, sampler=SequentialSampler(dataset),
                        collate_fn=collator, drop_last=False, num_workers=0)
    embs = np.zeros((len(dataset), configs["embsize"]), dtype=np.float32)
    count = 0
    with torch.no_grad():
        for batch in loader:
            gene_tokens = batch["gene"].to(device)
            out = model._encode(gene_tokens, batch["expr"].to(device),
                                src_key_padding_mask=gene_tokens.eq(pad_token_id))
            cls = out[:, 0, :].cpu().numpy()
            embs[count:count + len(cls)] = cls
            count += len(cls)
    return embs / np.maximum(np.linalg.norm(embs, axis=1, keepdims=True), 1e-12)


Z_train = embed(X_mouse, ids_mouse)[train]
Z_test = embed(X_human, ids_human)[test]
ytr, yte = y_mouse[train], y_human[test]

knn = KNeighborsClassifier(n_neighbors=15, metric="euclidean", weights="distance")
knn.fit(Z_train, ytr)
P_knn = bc.restrict_to_eval(bc.reorder_proba(knn.classes_, knn.predict_proba(Z_test)))

clf = LogisticRegression(max_iter=1000, solver="lbfgs")
clf.fit(Z_train, ytr)
P_lr = bc.restrict_to_eval(bc.reorder_proba(clf.classes_, clf.predict_proba(Z_test)))

bc.write_results("scgpt.csv", [
    {"Method": "scGPT + KNN (transductive)", **bc.family_metrics(yte, P_knn)},
    {"Method": "scGPT + LR (inductive)", **bc.family_metrics(yte, P_lr)},
])
