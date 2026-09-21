#!/usr/bin/env python3
"""Score six scikit-learn classifiers, PCA KNN, and Harmony KNN on the mouse to Kamath family transfer and write classical.csv.

Run: python run_classical.py
"""
import benchmark_common as bc

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import harmonypy as hm


def lognorm_rows(X):
    X = np.asarray(X, dtype=np.float64)
    tot = X.sum(axis=1, keepdims=True)
    tot[tot == 0] = 1.0
    X = X / tot * 1e4
    return np.log1p(X).astype(np.float32)


def prep_features(Xtr, Xte, scale):
    Xtr = lognorm_rows(Xtr)
    Xte = lognorm_rows(Xte)
    if scale:
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(Xtr)
        Xte = scaler.transform(Xte)
    return Xtr, Xte


def score_proba(classes, proba, yte):
    return bc.family_metrics(yte, bc.restrict_to_eval(bc.reorder_proba(classes, proba)))


def run_lr(Xtr, ytr, Xte, yte):
    Xtr, Xte = prep_features(Xtr, Xte, scale=True)
    clf = LogisticRegression(solver="lbfgs", max_iter=2000, C=1.0, random_state=bc.SEED)
    clf.fit(Xtr, ytr)
    return score_proba(clf.classes_, clf.predict_proba(Xte), yte)


def run_svm(Xtr, ytr, Xte, yte):
    Xtr, Xte = prep_features(Xtr, Xte, scale=True)
    clf = LinearSVC(C=1.0, max_iter=5000, random_state=bc.SEED, dual="auto")
    clf.fit(Xtr, ytr)
    dec = clf.decision_function(Xte)
    dec = dec - dec.max(axis=1, keepdims=True)
    proba = np.exp(dec)
    proba /= proba.sum(axis=1, keepdims=True)
    return score_proba(clf.classes_, proba, yte)


def run_rf(Xtr, ytr, Xte, yte):
    Xtr, Xte = prep_features(Xtr, Xte, scale=False)
    clf = RandomForestClassifier(n_estimators=200, random_state=bc.SEED, n_jobs=-1)
    clf.fit(Xtr, ytr)
    return score_proba(clf.classes_, clf.predict_proba(Xte), yte)


def run_cart(Xtr, ytr, Xte, yte):
    Xtr, Xte = prep_features(Xtr, Xte, scale=False)
    clf = DecisionTreeClassifier(random_state=bc.SEED)
    clf.fit(Xtr, ytr)
    return score_proba(clf.classes_, clf.predict_proba(Xte), yte)


def run_gnb(Xtr, ytr, Xte, yte):
    Xtr, Xte = prep_features(Xtr, Xte, scale=False)
    clf = GaussianNB()
    clf.fit(Xtr, ytr)
    return score_proba(clf.classes_, clf.predict_proba(Xte), yte)


def run_mlp(Xtr, ytr, Xte, yte):
    Xtr, Xte = prep_features(Xtr, Xte, scale=True)
    le = LabelEncoder()
    ytr_i = le.fit_transform(np.asarray(ytr))
    clf = MLPClassifier(hidden_layer_sizes=(256, 128), activation="relu", solver="adam",
                        alpha=1e-4, batch_size=512, learning_rate_init=1e-3, max_iter=200,
                        early_stopping=True, n_iter_no_change=10, random_state=bc.SEED)
    clf.fit(Xtr, ytr_i)
    labels = le.inverse_transform(np.asarray(clf.classes_).astype(int))
    return score_proba(labels, clf.predict_proba(Xte), yte)


def joint_pca(Xtr, Xte):
    Xall = np.vstack([np.asarray(Xtr, np.float32), np.asarray(Xte, np.float32)])
    Xall = StandardScaler().fit_transform(lognorm_rows(Xall))
    npcs = min(50, Xall.shape[0] - 1, Xall.shape[1])
    return PCA(n_components=npcs, random_state=bc.SEED).fit_transform(Xall)


def knn_score(Z, ntr, ytr, yte, k=15):
    knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean", weights="distance")
    knn.fit(Z[:ntr], ytr)
    return score_proba(knn.classes_, knn.predict_proba(Z[ntr:]), yte)


def run_knn(Xtr, ytr, Xte, yte):
    return knn_score(joint_pca(Xtr, Xte), len(Xtr), ytr, yte)


def run_harmony(Xtr, ytr, Xte, yte):
    ntr = len(Xtr)
    Xpca = joint_pca(Xtr, Xte).astype(np.float32)
    meta = pd.DataFrame({"batch": np.array(["mouse"] * ntr + ["human"] * (len(Xpca) - ntr))})
    ho = hm.run_harmony(Xpca, meta, "batch", max_iter_harmony=20, random_state=bc.SEED)
    Z = np.asarray(ho.Z_corr, np.float32)
    assert Z.shape == Xpca.shape
    return knn_score(Z, ntr, ytr, yte)


METHODS = {
    "Logistic Regression": run_lr,
    "Linear SVM": run_svm,
    "Random Forest": run_rf,
    "Decision Tree (CART)": run_cart,
    "Gaussian NB": run_gnb,
    "MLP": run_mlp,
    "KNN (k=15, PCA, transductive)": run_knn,
    "Harmony + KNN (transductive)": run_harmony,
}

Xtr, ytr, Xte, yte, _ = bc.load_features()
assert Xte.shape == (22048, bc.N_FEATURES)
rows = [{"Method": name, **fn(Xtr, ytr, Xte, yte)} for name, fn in METHODS.items()]
bc.write_results("classical.csv", rows)
