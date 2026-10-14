import logging
import os
import sys
from itertools import product

import numpy as np
import pkg_resources
import torch
import tqdm
from Bio.SeqIO.FastaIO import SimpleFastaParser
from numba import njit
from skorch import NeuralNetClassifier
from torch import nn

from tiara.src.transformations import TfidfWeighter, Transformer

logger = logging.getLogger(__name__)


# replace with your real data
DATA_DIR = sys.argv[1]
OUTPUT_DIR = sys.argv[2]
os.makedirs(OUTPUT_DIR, exist_ok=True)

torch.set_num_threads(int(sys.argv[3]))

first_stage_params = [
    dict(k=4, hidden_1=2048, hidden_2=2048, lr=0.001, dropout=0.2, epochs=41),
    dict(k=5, hidden_1=2048, hidden_2=2048, lr=0.001, dropout=0.2, epochs=28),
    dict(k=6, hidden_1=2048, hidden_2=1024, lr=0.001, dropout=0.2, epochs=41),
]

second_stage_params = [
    dict(k=4, hidden_1=256, hidden_2=128, lr=0.001, dropout=0.2, epochs=45),
    dict(k=5, hidden_1=256, hidden_2=128, lr=0.001, dropout=0.2, epochs=37),
    dict(k=6, hidden_1=256, hidden_2=128, lr=0.001, dropout=0.5, epochs=30),
    dict(k=7, hidden_1=128, hidden_2=64, lr=0.01, dropout=0.2, epochs=47),
]

kmer_to_pos_mappings = {k: {"".join(kmer): i for i, kmer in enumerate(product("ACGT", repeat=k))} for k in range(1, 12)}


@njit
def add_ones_matrix(mat, row, positions):
    for j in range(len(positions)):
        mat[row, positions[j]] += 1


def get_tfidf_repr(seqs, k, idf_vec):
    length = len(seqs)
    result = np.zeros((length, 4**k), dtype=np.float32)
    for i, seq in tqdm.tqdm(enumerate(seqs), total=len(seqs)):
        arr = []
        for pos in range(len(seq) - k + 1):
            try:
                arr.append(kmer_to_pos_mappings[k][seq[pos : pos + k]])
            except KeyError:
                continue
        positions = np.array(arr, dtype=np.int32)
        add_ones_matrix(result, i, positions)
    return result * idf_vec


logger.info("Started importing data")
with open(os.path.join(DATA_DIR, "mitochondria_fr.fasta")) as handle:
    mitochondria = [seq for _, seq in SimpleFastaParser(handle)]
with open(os.path.join(DATA_DIR, "plast_fr.fasta")) as handle:
    plastids = [seq for _, seq in SimpleFastaParser(handle)]
with open(os.path.join(DATA_DIR, "bacteria_fr.fasta")) as handle:
    bacteria = [seq for _, seq in SimpleFastaParser(handle)]
with open(os.path.join(DATA_DIR, "eukarya_fr.fasta")) as handle:
    eukarya = [seq.upper() for _, seq in SimpleFastaParser(handle) if set(seq.upper()) == set("ACGT")]
with open(os.path.join(DATA_DIR, "archaea_fr.fasta")) as handle:
    archea = [seq for _, seq in SimpleFastaParser(handle)]


def get_data(stage, k):
    transformer = Transformer(
        TfidfWeighter.load_params(
            pkg_resources.resource_filename(__name__, f"../models/tfidf-models/k{k}-{stage}-stage")
        ),
        fragment_len=5000,
        k=k,
    )
    transformer.model.idfs = transformer.model.idfs.astype(np.float32)
    logger.info(f"Computing representations for stage {stage} and kmer {k}")
    if stage == "first":
        X = get_tfidf_repr(
            plastids + mitochondria + bacteria + archea + eukarya,
            k,
            transformer.model.idfs,
        )
        y = np.array(
            [0] * (len(plastids) + len(mitochondria)) + [1] * len(bacteria) + [3] * len(archea) + [4] * len(eukarya)
        )
    elif stage == "second":
        X = get_tfidf_repr(plastids + mitochondria, k, transformer.model.idfs)
        y = np.array([0] * len(plastids) + [2] * len(mitochondria))
    else:
        raise ValueError("Wrong stage!")
    X = X / np.linalg.norm(X, axis=1).reshape((-1, 1))
    logger.info("Done")
    return X, y


class MyNNet_2(nn.Sequential):
    def __init__(self, dim_in, hid1, hid2, dim_out, dropout):
        super().__init__(
            nn.Linear(dim_in, hid1),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(hid1, hid2),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(hid2, dim_out),
            nn.Softmax(1),
        )


for arch in first_stage_params:
    logger.info(f"Training architecture {arch}")
    k = arch["k"]
    X, y = get_data("first", k)
    net = NeuralNetClassifier(
        MyNNet_2(4 ** arch["k"], arch["hidden_1"], arch["hidden_2"], 5, arch["dropout"]),
        max_epochs=arch["epochs"],
        lr=arch["lr"],
        train_split=None,
        iterator_train__shuffle=True,
        optimizer=torch.optim.Adam,
        verbose=10,
    )
    net.fit(X, y)
    net.save_params(
        f_params=pkg_resources.resource_filename(
            __name__,
            os.path.join(OUTPUT_DIR, "_".join([f"{k}-{v}" for k, v in arch.items()])) + ".pkl",
        )
    )

for arch in second_stage_params:
    logger.info(f"Training architecture {arch}")
    k = arch["k"]
    X, y = get_data("second", k)
    net = NeuralNetClassifier(
        MyNNet_2(4 ** arch["k"], arch["hidden_1"], arch["hidden_2"], 3, arch["dropout"]),
        max_epochs=arch["epochs"],
        lr=arch["lr"],
        train_split=None,
        iterator_train__shuffle=True,
        optimizer=torch.optim.Adam,
        verbose=10,
    )
    net.fit(X, y)
    net.save_params(
        f_params=pkg_resources.resource_filename(
            __name__,
            os.path.join(OUTPUT_DIR, "_".join([f"{k}-{v}" for k, v in arch.items()])) + ".pkl",
        )
    )
