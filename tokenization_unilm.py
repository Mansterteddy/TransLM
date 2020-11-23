import os
import sentencepiece
import collections

def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    index = 0
    with open(vocab_file, "r", encoding="utf-8") as reader:
        while True:
            token = reader.readline()
            if not token:
                break
            token = token.rstrip()
            if token in vocab.keys():
                raise RuntimeError("redundent entry in vocab")
            vocab[token] = index
            index += 1
    return vocab

class SPETokenizer(object):
    """Runs end-to-end tokenization: punctuation splitting + wordpiece"""

    def __init__(self, vocab_file, bpe_model_file, max_len=None):
        if not os.path.isfile(vocab_file):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'. To load the vocabulary from a Google pretrained "
                "model use `tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`".format(vocab_file))
        self.vocab = load_vocab(vocab_file)
        self.ids_to_tokens = collections.OrderedDict(
            [(ids, tok) for tok, ids in self.vocab.items()])
        self.sp_tokenizer = sentencepiece.SentencePieceProcessor()
        self.sp_tokenizer.Load(bpe_model_file)
        self.max_len = max_len if max_len is not None else int(1e12)

    def vocab_size(self):
        return len(self.vocab)

    def tokenize(self, text):
        split_tokens = ["[UNK]" if x not in self.vocab else x for x in self.sp_tokenizer.EncodeAsPieces(text)]
        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        """Converts a sequence of tokens into ids using the vocab."""
        ids = []
        for token in tokens:
            ids.append(self.vocab[token])
        # This is quite atrocious. Max length should not be handled by tokenizer.
        #if len(ids) > self.max_len:
        #    raise ValueError(
        #        "Token indices sequence length is longer than the specified maximum "
        #        " sequence length for this BERT model ({} > {}). Running this"
        #        " sequence through BERT will result in indexing errors".format(
        #            len(ids), self.max_len)
        #    )
        return ids

    def convert_ids_to_tokens(self, ids):
        """Converts a sequence of ids in wordpiece tokens using the vocab."""
        tokens = []
        for i in ids:
            tokens.append(self.ids_to_tokens[i])
        return tokens

    @classmethod
    def from_pretrained(cls, pretrained_model_path):
        """
        Instantiate a PreTrainedBertModel from a pre-trained model file.
        Download and cache the pre-trained model file if needed.
        """
        vocab_file = os.path.join(pretrained_model_path, "vocab.txt")
        bpe_file = os.path.join(pretrained_model_path, "sentencepiece.bpe.model")
        tokenizer = cls(vocab_file, bpe_file)
        return tokenizer
