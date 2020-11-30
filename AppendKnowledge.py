import re
import unicodedata
from urllib.parse import unquote

import transformers
import tokenization_unilm

import numpy as np
import torch 
import torch.autograd as autograd
import torch.nn.functional as F
from torch import nn

from transformers.modeling_bert import BertModel, BertPreTrainedModel

class RankLMV5(BertPreTrainedModel):
    def __init__(self, config):
        super(RankLMV5, self).__init__(config)
        self.bert = BertModel(config)
        self.cls_hrs = nn.Sequential(*self.default_head(1024, 5))
        self.init_weights()
    
    def default_head(self, hidden_size, classes, use_numfeat=False):
        return [nn.Linear(2 * hidden_size if use_numfeat else hidden_size, hidden_size),
                nn.Tanh(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size, classes)]

    def forward(self, input_ids, token_type_ids, attention_mask):

        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        cls_output = outputs[0][:,0,:]
        hrs_logits = self.cls_hrs(cls_output)

        softmax_layer = nn.Softmax(dim=1)
        hrs_scores = softmax_layer(hrs_logits)

        return hrs_scores

class EvalModel(object):

    def __init__(self, cuda=True):
        self.tokenizer = tokenization_unilm.SPETokenizer("./BingLR_Vocab/vocab.txt", "./BingLR_Vocab/sentencepiece.bpe.model")
        print(self.tokenizer.vocab_size())
        
        self.market_list = ["lang_de", "lang_en", "lang_es", "lang_fr", "lang_it", "lang_ja", "lang_pt", "lang_zh", "lang_un", "dist_de", "dist_au", "dist_ca", "dist_gb", "dist_in", "dist_us", "dist_es", "dist_mx", "dist_fr", "dist_it", "dist_jp", "dist_br", "dist_cn", "dist_hk", "dist_tw", "dist_un"]
        self.pattern = re.compile(r'<[^>]+>', re.S)
        self.max_seq_length = 128

        self.to_cuda = lambda x: x.to("cuda") if cuda else x

        self.model_path = "./BingLR_Model/"
        binglr_state_dict = torch.load("./BingLR_Model/pytorch_model.bin", map_location='cpu')["model"]
        self.model = self.to_cuda(RankLMV5.from_pretrained(self.model_path, state_dict=binglr_state_dict))
        self.model.eval()

    def _truncate_seq(self, tokens, max_length):
        """Truncates a sequence pair in place to the maximum length."""
        while True:
            if len(tokens) <= max_length:
                break
            tokens.pop()
        return tokens

    def _truncate_seq_pair(self, tokens_a, tokens_b, max_length, max_query_length=32):
        """Truncates a sequence pair in place to the maximum length."""
        # Only pop b
        while True:
            while len(tokens_a) > max_query_length:
                tokens_a.pop()
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            tokens_b.pop()

    def NormalizedUrl(self, url):
        url = url.lower().strip().replace("http://", "").replace("https://", "")
        if url[0:4] == "www.":
            url = url[4:]
        sharpIndex = url.find('#')
        if sharpIndex > 0:
            url = url[0:sharpIndex]
        while len(url)>0 and url[-1] == "/":
            url = url.rstrip("/")
        return url

    def remove_control_char(self, input_string):
        return ''.join([c for c in input_string if not unicodedata.category(c).startswith('C')])

    def tokenize(self, query, query_aug, url, title, snippet, market):
        
        query = self.remove_control_char(query.lower().strip())
        query_aug = self.remove_control_char(query_aug.lower().strip())
        title = title.lower().replace("\ue000", "").replace("\ue001", "").replace("...", "")
        snippet = snippet.lower().replace("\ue000", "").replace("\ue001", "")

        title = self.remove_control_char(self.pattern.sub('', title))
        snippet = self.remove_control_char(self.pattern.sub('', snippet))

        url = url.replace("\/", "/")
        url = unquote(url)
        url = re.sub(r'[^\w\s\/]', ' ', self.NormalizedUrl(url), re.UNICODE)
        url = self.tokenizer.tokenize(url)

        market = market.lower()      
        language = "lang_" + market.split('-')[0] if len(market.split('-')) == 2 else "un"
        district = "dist_" + market.split('-')[1] if len(market.split('-')) == 2 else "un"

        if language not in self.market_list:
            language = "lang_un"
        if district not in self.market_list:
            district = "dist_un"
               
        query = self.tokenizer.tokenize(query)
        query_aug = self.tokenizer.tokenize(query_aug)
        title = self.tokenizer.tokenize(title)
        snippet = self.tokenizer.tokenize(snippet)

        query = self._truncate_seq(query, 16)
        query_aug = self._truncate_seq(query_aug, 16)
        url = self._truncate_seq(url, 32)
        title = self._truncate_seq(title, 32)

        query = query + query_aug + [language, district]

        doc = [["[Title]"], title, ["[Url]"], url, ["[Snippet]"], snippet]
        doc = [x for y in doc for x in y]
        self._truncate_seq_pair(query, doc, self.max_seq_length - 3)

        tokens = [["[CLS]"], query, ["[SEP]"], doc, ["[SEP]"]]
        tokens = [x for y in tokens for x in y]

        actual_size = len(tokens)
        query_length = len(query)

        token_index_list = []
        for i in range(len(tokens)):
            token_index_list.append((i, tokens[i]))
        print(token_index_list)

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * actual_size

        segment_ids = [[0] * (query_length + 2), [1] * (len(doc) + 1)]
        segment_ids = [x for y in segment_ids for x in y]

        return input_ids, segment_ids, input_mask

    def inference_test(self, input_ids, input_mask, segment_ids):
        
        input_ids = self.to_cuda(torch.tensor([input_ids]))
        input_mask = self.to_cuda(torch.tensor([input_mask]))
        segment_ids = self.to_cuda(torch.tensor([segment_ids]))

        output = self.model(input_ids, input_mask, segment_ids)
        output = output.detach().cpu().numpy()

        output = output[0][0] * 1 + output[0][1] * 2 + output[0][2] * 3 + output[0][3] * 4 + output[0][4] * 5
        print(output)


# Init instance
masklm_ins = EvalModel()

# Test Case

query = "how much did microsoft pay for lobe"
query_aug = "In 2018, Microsoft bought Lobe, a San Francisco-based startup that made a platform for building, training and shipping custom deep-learning models. This week, Microsoft made some of Lobe's technology publicly available. Load Error. On October 26, available a public preview of a Lobe app for training machine-learning models."
url = "https://en.wikipedia.org/wiki/List_of_mergers_and_acquisitions_by_Microsoft"
title = "List of mergers and acquisitions by Microsoft - Wikipedia"
snippet = "Key acquisitions. Microsoft's first acquisition was Forethought on July 30, 1987. Forethought was founded in 1983 and developed a presentation program that would later be known as Microsoft PowerPoint.. On December 31, 1997, Microsoft acquired Hotmail.com for $500 million, its largest acquisition at the time, and integrated Hotmail into its MSN group of services."
market = "en-us"

input_ids, segment_ids, input_mask = masklm_ins.tokenize(query, query_aug, url, title, snippet, market)
masklm_ins.inference_test(input_ids, input_mask, segment_ids)

query = "how much did microsoft pay for lobe"
query_aug = "In 2018, Microsoft bought Lobe, a San Francisco-based startup that made a platform for building, training and shipping custom deep-learning models. This week, Microsoft made some of Lobe's technology publicly available. Load Error. On October 26, available a public preview of a Lobe app for training machine-learning models."
url = "https://www.cnbc.com/2018/09/13/microsoft-acquires-lobe-ai-startup.html"
title = "Microsoft acquires Lobe, A.I start-up - CNBC"
snippet = "Microsoft buys Lobe, a small start-up that makes it easier to build A.I. apps. Published Thu, Sep 13 2018 1:53 PM EDT. Jordan Novet @jordannovet. Key Points."
market = "en-us"

input_ids, segment_ids, input_mask = masklm_ins.tokenize(query, query_aug, url, title, snippet, market)
masklm_ins.inference_test(input_ids, input_mask, segment_ids)