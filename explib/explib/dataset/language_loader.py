import os, sys
import glob

from collections import Counter, OrderedDict
import numpy as np
import torch
import subprocess

# Code copied from https://github.com/kimiyoung/transformer-xl
from explib import config


class Vocab(object):
    def __init__(
        self,
        special=[],
        min_freq=0,
        max_size=None,
        lower_case=True,
        delimiter=None,
        vocab_file=None,
    ):
        self.counter = Counter()
        self.special = special
        self.min_freq = min_freq
        self.max_size = max_size
        self.lower_case = lower_case
        self.delimiter = delimiter
        self.vocab_file = vocab_file

    def tokenize(self, line, add_eos=False, add_double_eos=False):
        line = line.strip()
        # convert to lower case
        if self.lower_case:
            line = line.lower()

        # empty delimiter '' will evaluate False
        if self.delimiter == "":
            symbols = line
        else:
            symbols = line.split(self.delimiter)

        if add_double_eos:  # lm1b
            return ["<S>"] + symbols + ["<S>"]
        elif add_eos:
            return symbols + ["<eos>"]
        else:
            return symbols

    def count_file(self, path, verbose=False, add_eos=False):
        if verbose:
            print("counting file {} ...".format(path))
        assert os.path.exists(path)

        sents = []
        with open(path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if verbose and idx > 0 and idx % 500000 == 0:
                    print("    line {}".format(idx))
                symbols = self.tokenize(line, add_eos=add_eos)
                self.counter.update(symbols)
                sents.append(symbols)

        return sents

    def count_sents(self, sents, verbose=False):
        """
        sents : a list of sentences, each a list of tokenized symbols
        """
        if verbose:
            print("counting {} sents ...".format(len(sents)))
        for idx, symbols in enumerate(sents):
            if verbose and idx > 0 and idx % 500000 == 0:
                print("    line {}".format(idx))
            self.counter.update(symbols)

    def _build_from_file(self, vocab_file):
        self.idx2sym = []
        self.sym2idx = OrderedDict()

        with open(vocab_file, "r", encoding="utf-8") as f:
            for line in f:
                symb = line.strip().split()[0]
                self.add_symbol(symb)
        self.unk_idx = self.sym2idx["<UNK>"]

    def build_vocab(self):
        if self.vocab_file:
            print("building vocab from {}".format(self.vocab_file))
            self._build_from_file(self.vocab_file)
            print("final vocab size {}".format(len(self)))
        else:
            print(
                "building vocab with min_freq={}, max_size={}".format(
                    self.min_freq, self.max_size
                )
            )
            self.idx2sym = []
            self.sym2idx = OrderedDict()

            for sym in self.special:
                self.add_special(sym)

            for sym, cnt in self.counter.most_common(self.max_size):
                if cnt < self.min_freq:
                    break
                self.add_symbol(sym)

            print(
                "final vocab size {} from {} unique tokens".format(
                    len(self), len(self.counter)
                )
            )

    def encode_file(
        self, path, ordered=False, verbose=False, add_eos=True, add_double_eos=False
    ):
        if verbose:
            print("encoding file {} ...".format(path))
        assert os.path.exists(path)
        encoded = []
        with open(path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if verbose and idx > 0 and idx % 500000 == 0:
                    print("    line {}".format(idx))
                symbols = self.tokenize(
                    line, add_eos=add_eos, add_double_eos=add_double_eos
                )
                encoded.append(self.convert_to_tensor(symbols))

        if ordered:
            encoded = torch.cat(encoded)

        return encoded

    def encode_sents(self, sents, ordered=False, verbose=False):
        if verbose:
            print("encoding {} sents ...".format(len(sents)))
        encoded = []
        for idx, symbols in enumerate(sents):
            if verbose and idx > 0 and idx % 500000 == 0:
                print("    line {}".format(idx))
            encoded.append(self.convert_to_tensor(symbols))

        if ordered:
            encoded = torch.cat(encoded)

        return encoded

    def add_special(self, sym):
        if sym not in self.sym2idx:
            self.idx2sym.append(sym)
            self.sym2idx[sym] = len(self.idx2sym) - 1
            setattr(self, "{}_idx".format(sym.strip("<>")), self.sym2idx[sym])

    def add_symbol(self, sym):
        if sym not in self.sym2idx:
            self.idx2sym.append(sym)
            self.sym2idx[sym] = len(self.idx2sym) - 1

    def get_sym(self, idx):
        assert 0 <= idx < len(self), "Index {} out of range".format(idx)
        return self.idx2sym[idx]

    def get_idx(self, sym):
        if sym in self.sym2idx:
            return self.sym2idx[sym]
        else:
            # print('encounter unk {}'.format(sym))
            assert "<eos>" not in sym
            assert hasattr(self, "unk_idx")
            return self.sym2idx.get(sym, self.unk_idx)

    def get_symbols(self, indices):
        return [self.get_sym(idx) for idx in indices]

    def get_indices(self, symbols):
        return [self.get_idx(sym) for sym in symbols]

    def convert_to_tensor(self, symbols):
        return torch.LongTensor(self.get_indices(symbols))

    def convert_to_sent(self, indices, exclude=None):
        if exclude is None:
            return " ".join([self.get_sym(idx) for idx in indices])
        else:
            return " ".join(
                [self.get_sym(idx) for idx in indices if idx not in exclude]
            )

    def convert_to_sent_from_tensor(self, indices):
        sents = []
        for sent in indices:
            sents.append(" ".join([self.get_sym(int(idx)) for idx in sent]))
        return sents

    def __len__(self):
        return len(self.idx2sym)


class LMOrderedIterator(object):
    def __init__(
        self,
        data,
        bsz,
        bptt,
        device="cpu",
        ext_len=None,
        drop_last=False,
        outliers_filename=None,
    ):
        """
        data -- LongTensor -- the LongTensor is strictly ordered
        """
        self.bsz = bsz
        self.bptt = bptt
        self.ext_len = ext_len if ext_len is not None else 0

        self.device = device

        # Work out how cleanly we can divide the dataset into bsz parts.
        self.n_step = data.size(0) // bsz

        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, self.n_step * bsz)

        if outliers_filename is not None:
            outlier_indices = np.load(
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "outliers",
                    outliers_filename,
                )
            )
            outlier_indices = sorted(
                list(map(int, np.ndarray.tolist(outlier_indices))), reverse=True
            )
            for idx in outlier_indices:
                data = torch.cat(
                    [data[0 : idx * self.bptt], data[(idx + 1) * self.bptt :]]
                )
            self.n_step = data.size(0) // bsz

        # Evenly divide the data across the bsz batches.
        self.data = data.view(bsz, -1).t().contiguous().to(device)

        # Number of mini-batches
        self.n_batch = (self.n_step + self.bptt - 1) // self.bptt

        self.drop_last = drop_last

        if self.drop_last and (self.n_step + self.bptt - 1) % self.bptt != 0:
            self.n_batch = self.n_batch - 1

    def __len__(self):
        return self.n_batch

    def get_batch(self, i, bptt=None):
        if bptt is None:
            bptt = self.bptt
        seq_len = min(bptt, self.data.size(0) - 1 - i)

        end_idx = i + seq_len
        beg_idx = max(0, i - self.ext_len)

        data = self.data[beg_idx:end_idx]
        target = self.data[i + 1 : i + 1 + seq_len]

        return data, target, seq_len

    def get_fixlen_iter(self, start=0):
        end = self.data.size(0) - 1
        if self.drop_last:
            end = self.data.size(0) - 1 - ((self.data.size(0) - 1) % self.bptt)
        for i in range(start, end, self.bptt):
            yield self.get_batch(i)

    def get_varlen_iter(self, start=0, std=5, min_len=5, max_deviation=3):
        max_len = self.bptt + max_deviation * std
        i = start
        while True:
            bptt = self.bptt if np.random.random() < 0.95 else self.bptt / 2.0
            bptt = min(max_len, max(min_len, int(np.random.normal(bptt, std))))
            data, target, seq_len = self.get_batch(i, bptt)
            i += seq_len
            yield data, target, seq_len
            if i >= self.data.size(0) - 2:
                break

    def __iter__(self):
        return self.get_fixlen_iter()


class LMShuffledIterator(object):
    def __init__(self, data, bsz, bptt, device="cpu", ext_len=None, shuffle=False):
        """
        data -- list[LongTensor] -- there is no order among the LongTensors
        """
        self.data = data

        self.bsz = bsz
        self.bptt = bptt
        self.ext_len = ext_len if ext_len is not None else 0

        self.device = device
        self.shuffle = shuffle

    def get_sent_stream(self):
        # index iterator
        epoch_indices = (
            np.random.permutation(len(self.data))
            if self.shuffle
            else np.array(range(len(self.data)))
        )

        # sentence iterator
        for idx in epoch_indices:
            yield self.data[idx]

    def stream_iterator(self, sent_stream):
        # streams for each data in the batch
        streams = [None] * self.bsz

        data = torch.LongTensor(self.bptt, self.bsz)
        target = torch.LongTensor(self.bptt, self.bsz)

        n_retain = 0

        while True:
            # data   : [n_retain+bptt x bsz]
            # target : [bptt x bsz]
            data[n_retain:].fill_(-1)
            target.fill_(-1)

            valid_batch = True

            for i in range(self.bsz):
                n_filled = 0
                try:
                    while n_filled < self.bptt:
                        if streams[i] is None or len(streams[i]) <= 1:
                            streams[i] = next(sent_stream)
                        # number of new tokens to fill in
                        n_new = min(len(streams[i]) - 1, self.bptt - n_filled)
                        # first n_retain tokens are retained from last batch
                        data[
                            n_retain + n_filled : n_retain + n_filled + n_new, i
                        ] = streams[i][:n_new]
                        target[n_filled : n_filled + n_new, i] = streams[i][
                            1 : n_new + 1
                        ]
                        streams[i] = streams[i][n_new:]
                        n_filled += n_new
                except StopIteration:
                    valid_batch = False
                    break

            if not valid_batch:
                return

            data = data.to(self.device)
            target = target.to(self.device)

            yield data, target, self.bptt

            n_retain = min(data.size(0), self.ext_len)
            if n_retain > 0:
                data[:n_retain] = data[-n_retain:]
            data.resize_(n_retain + self.bptt, data.size(1))

    def __iter__(self):
        # sent_stream is an iterator
        sent_stream = self.get_sent_stream()

        for batch in self.stream_iterator(sent_stream):
            yield batch


class Corpus(object):
    def __init__(self, path, dataset, *args, **kwargs):
        self.dataset = dataset
        self.vocab = Vocab(*args, **kwargs)

        if self.dataset in ["ptb", "wt2", "enwik8", "text8"]:
            self.vocab.count_file(os.path.join(path, "train.txt"))
            self.vocab.count_file(os.path.join(path, "valid.txt"))
            self.vocab.count_file(os.path.join(path, "test.txt"))
        elif self.dataset == "wt103":
            self.vocab.count_file(os.path.join(path, "train.txt"))
        elif self.dataset == "lm1b":
            train_path_pattern = os.path.join(
                path,
                "1-billion-word-language-modeling-benchmark-r13output",
                "training-monolingual.tokenized.shuffled",
                "news.en-*",
            )
            train_paths = glob.glob(train_path_pattern)
            # the vocab will load from file when build_vocab() is called

        self.vocab.build_vocab()

        if self.dataset in ["ptb", "wt2", "wt103"]:
            self.train = self.vocab.encode_file(
                os.path.join(path, "train.txt"), ordered=True
            )
            self.valid = self.vocab.encode_file(
                os.path.join(path, "valid.txt"), ordered=True
            )
            self.test = self.vocab.encode_file(
                os.path.join(path, "test.txt"), ordered=True
            )
        elif self.dataset in ["enwik8", "text8"]:
            self.train = self.vocab.encode_file(
                os.path.join(path, "train.txt"), ordered=True, add_eos=False
            )
            self.valid = self.vocab.encode_file(
                os.path.join(path, "valid.txt"), ordered=True, add_eos=False
            )
            self.test = self.vocab.encode_file(
                os.path.join(path, "test.txt"), ordered=True, add_eos=False
            )
        elif self.dataset == "lm1b":
            self.train = train_paths
            self.valid = self.vocab.encode_file(
                os.path.join(path, "valid.txt"), ordered=False, add_double_eos=True
            )
            self.test = self.vocab.encode_file(
                os.path.join(path, "test.txt"), ordered=False, add_double_eos=True
            )

    def get_iterator(self, split, *args, **kwargs):
        if split == "train":
            if self.dataset in ["ptb", "wt2", "wt103", "enwik8", "text8"]:
                data_iter = LMOrderedIterator(self.train, *args, **kwargs)
        elif split in ["valid", "test"]:
            data = self.valid if split == "valid" else self.test
            if self.dataset in ["ptb", "wt2", "wt103", "enwik8", "text8"]:
                data_iter = LMOrderedIterator(data, *args, **kwargs)

        return data_iter


def get_lm_corpus(datadir, dataset):
    fn = os.path.join(datadir, "cache.pt")
    if os.path.exists(fn):
        print("Loading cached dataset...")
        corpus = torch.load(fn)
    else:
        print("Producing dataset {}...".format(dataset))
        kwargs = {}
        if dataset in ["wt103", "wt2"]:
            kwargs["special"] = ["<eos>"]
            kwargs["lower_case"] = False
        elif dataset == "ptb":
            kwargs["special"] = ["<eos>"]
            kwargs["lower_case"] = True
        elif dataset == "lm1b":
            kwargs["special"] = []
            kwargs["lower_case"] = False
            kwargs["vocab_file"] = os.path.join(datadir, "1b_word_vocab.txt")
        elif dataset in ["enwik8", "text8"]:
            pass

        corpus = Corpus(datadir, dataset, **kwargs)
        torch.save(corpus, fn)

    return corpus


def ptb_loader(
    batch_size,
    device,
    tgt_len,
    drop_last=False,
    outliers_filename=None,
):
    datadir = os.path.join(config.get_workspace(), "datasets", "penn")
    cwd = os.path.dirname(os.path.realpath(__file__))
    if not os.path.isdir(datadir):
        result = subprocess.run(
            [
                "sh",
                "./get_ptb.sh",
                os.path.abspath(os.path.join(config.get_workspace(), "datasets")),
            ],
            check=True,
            cwd=cwd,
            capture_output=True,
            text=True,
        )
        print("Shell get_ptb.sh: stdout")
        print(result.stdout)
        print("Shell get_ptb.sh: stderr")
        print(result.stderr)

    corpus = get_lm_corpus(datadir, "ptb")
    ntokens = len(corpus.vocab)
    tr_iter = corpus.get_iterator(
        "train",
        batch_size,
        tgt_len,
        device=device,
        ext_len=0,
        drop_last=drop_last,
        outliers_filename=outliers_filename,
    )

    te_iter = corpus.get_iterator("test", batch_size, tgt_len, device=device, ext_len=0)
    return tr_iter, te_iter, ntokens


def wikitext2_loader(
    batch_size,
    device,
    tgt_len,
    drop_last=False,
):
    datadir = os.path.join(config.get_workspace(), "datasets", "wikitext-2")
    cwd = os.path.dirname(os.path.realpath(__file__))
    if not os.path.isdir(datadir):
        result = subprocess.run(
            [
                "sh",
                "./get_wikitext2.sh",
                os.path.abspath(os.path.join(config.get_workspace(), "datasets")),
            ],
            check=True,
            cwd=cwd,
            capture_output=True,
            text=True,
        )
        print("Shell get_wikitext2.sh: stdout")
        print(result.stdout)
        print("Shell get_wikitext2.sh: stderr")
        print(result.stderr)

    corpus = get_lm_corpus(datadir, "wt2")
    ntokens = len(corpus.vocab)
    tr_iter = corpus.get_iterator(
        "train",
        batch_size,
        tgt_len,
        device=device,
        ext_len=0,
        drop_last=drop_last,
    )
    te_iter = corpus.get_iterator("test", batch_size, tgt_len, device=device, ext_len=0)
    return tr_iter, te_iter, ntokens, corpus
