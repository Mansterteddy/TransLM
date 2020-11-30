import re
import unicodedata
from urllib.parse import unquote

import transformers
import tokenization_unilm

import numpy as np
import torch 
import torch.autograd as autograd
import torch.nn.functional as F

class EvalMaskLM(object):

    def __init__(self, cuda=True):
        self.tokenizer = tokenization_unilm.SPETokenizer("./BingLR_Vocab/vocab.txt", "./BingLR_Vocab/sentencepiece.bpe.model")
        print(self.tokenizer.vocab_size())
        
        self.market_list = ["lang_de", "lang_en", "lang_es", "lang_fr", "lang_it", "lang_ja", "lang_pt", "lang_zh", "lang_un", "dist_de", "dist_au", "dist_ca", "dist_gb", "dist_in", "dist_us", "dist_es", "dist_mx", "dist_fr", "dist_it", "dist_jp", "dist_br", "dist_cn", "dist_hk", "dist_tw", "dist_un"]
        self.pattern = re.compile(r'<[^>]+>', re.S)
        self.max_seq_length = 128

        self.to_cuda = lambda x: x.to("cuda") if cuda else x

        self.model_path = "./BingLR_Model/"
        binglr_state_dict = torch.load("./BingLR_Model/pytorch_model.bin", map_location='cpu')["model"]
        self.model = self.to_cuda(transformers.BertForMaskedLM.from_pretrained(self.model_path, state_dict=binglr_state_dict))
        self.model.eval()

        self.topk = 20
        self.mask_id = 250001

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

    def tokenize(self, query, url, title, snippet, market):
        
        query = self.remove_control_char(query.lower().strip())
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
        title = self.tokenizer.tokenize(title)
        snippet = self.tokenizer.tokenize(snippet)

        query = self._truncate_seq(query, 16)
        url = self._truncate_seq(url, 32)
        title = self._truncate_seq(title, 32)

        query = query + [language, district]

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

    def inference_test(self, input_ids, input_mask, segment_ids, token_index):
        
        #print(input_ids)
        input_ids[token_index] = self.mask_id
        #print(input_ids)
        #assert 0 == 1

        input_ids = self.to_cuda(torch.tensor([input_ids]))
        input_mask = self.to_cuda(torch.tensor([input_mask]))
        segment_ids = self.to_cuda(torch.tensor([segment_ids]))

        output = self.model(input_ids, input_mask, segment_ids)

        cur_logits = autograd.Variable(output[0][token_index])
        cur_prob = F.softmax(cur_logits, dim=0)

        topk_id = torch.topk(cur_prob, self.topk)[1].tolist()
        #print("topk_id: ", topk_id)
        topk_token = self.tokenizer.convert_ids_to_tokens(topk_id)
        print("\n")
        print("topk_token: ", topk_token)
        print("\n")

    def inference_test_list(self, input_ids, input_mask, segment_ids, token_index, token_index_list):
        
        input_ids[token_index] = self.mask_id

        for index_item in token_index_list:
            input_ids[index_item] = self.mask_id

        input_ids = self.to_cuda(torch.tensor([input_ids]))
        input_mask = self.to_cuda(torch.tensor([input_mask]))
        segment_ids = self.to_cuda(torch.tensor([segment_ids]))

        output = self.model(input_ids, input_mask, segment_ids)

        cur_logits = autograd.Variable(output[0][token_index])
        cur_prob = F.softmax(cur_logits, dim=0)

        topk_id = torch.topk(cur_prob, self.topk)[1].tolist()
        #print("topk_id: ", topk_id)
        topk_token = self.tokenizer.convert_ids_to_tokens(topk_id)
        print("\n")
        print("topk_token: ", topk_token)
        print("\n")

# Init instance
masklm_ins = EvalMaskLM()

# Test Case

query = "how much did microsoft pay for lobe"
url = ""
title = ""
snippet = "Microsoft releases preview of Lobe training app for machine-learning ZDNet. Microsoft is continuing to look for ways to make machine-learning technology easier to use. In 2018, Microsoft bought Lobe, a San Francisco-based startup that made a platform for building, training and shipping custom deep-learning models."
market = "en-us"

input_ids, segment_ids, input_mask = masklm_ins.tokenize(query, url, title, snippet, market)
#print("input_ids: ", input_ids)

mask_list = [8, 9]
masklm_ins.inference_test(input_ids, input_mask, segment_ids, 8)
masklm_ins.inference_test(input_ids, input_mask, segment_ids, 9)