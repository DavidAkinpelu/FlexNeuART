#!/usr/bin/env python
#
#  Copyright 2014+ Carnegie Mellon University
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import torch

from typing import Optional, List

from transformers import PreTrainedTokenizerBase, BatchEncoding
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

from flexneuart import models
from flexneuart.models.base_decoder_only import DecoderBaseRanker
from flexneuart.models.utils import mean_pool

DEFAULT_DROPOUT = 0.1

DEFAULT_LORA_R = 32
DEFAULT_LORA_ALPHA = 64
DEFAULT_LORA_DROPOUT = 0.1
DEFAULT_LOAD_IN_4BIT = False
DEFAULT_LOAD_IN_8BIT = False
DEFAULT_LOAD_IN_2BIT = False
DEFAULT_TRUST_REMOTE_CODE = False
DEFAULT_USE_SEP = False
DEFAULT_LOFTQ_BITS = None
DEFAULT_LOFTQ_ITER = None
DEFAULT_USE_DORA = False
DEFAULT_QUANT_TYPE = None
DEFAULT_GALORE = False


@models.register("peft")
class DecoderRanker(DecoderBaseRanker):
    """
        A standard vanilla BERT Ranker, which does not pad queries (unlike CEDR version of FirstP).

        Nogueira, Rodrigo, and Kyunghyun Cho. "Passage Re-ranking with BERT."
        arXiv preprint arXiv:1901.04085 (2019).

    """
    def __init__(self, base_model_flavor, lora_target_modules: list=[], lora_r: int=DEFAULT_LORA_R, use_dora: bool=DEFAULT_USE_DORA,
                 lora_alpha: int=DEFAULT_LORA_ALPHA, lora_dropout: float=DEFAULT_LORA_DROPOUT, quant_type: Optional[str]=DEFAULT_QUANT_TYPE,
                 load_in_4bit: bool=DEFAULT_LOAD_IN_4BIT,
                 load_in_8bit: bool=DEFAULT_LOAD_IN_8BIT, trust_remote_code: bool=DEFAULT_TRUST_REMOTE_CODE, use_sep: bool=DEFAULT_USE_SEP, 
                 use_mean_pool: bool=False, dropout: float=DEFAULT_DROPOUT):
        
        super().__init__(base_model_flavor=base_model_flavor, quant_type=quant_type, lora_target_modules=lora_target_modules, lora_r=lora_r, 
                         lora_alpha=lora_alpha, lora_dropout=lora_dropout, use_dora=use_dora, load_in_4bit=load_in_4bit, load_in_8bit=load_in_8bit, 
                         trust_remote_code=trust_remote_code)
        self.use_sep = use_sep
        self.use_mean_pool = use_mean_pool
        self.dropout = torch.nn.Dropout(dropout)
        print('Dropout', self.dropout, 'use_mean_pool:', use_mean_pool)
        self.eos = torch.nn.Linear(self.model_dim, 1)
        torch.nn.init.xavier_uniform_(self.eos.weight)

    def featurize(self, max_query_len : int, max_doc_len : int,
                        query_texts : List[str],
                        doc_texts : List[str]) -> tuple:
        """
           "Featurizes" input. This function itself create a batch
            b/c training code does not use a standard PyTorch loader!
        """
        tok : PreTrainedTokenizerBase = self.tokenizer
        assert len(query_texts) == len(doc_texts), \
            "Document array length must be the same as query array length!"
        if self.use_sep:
            query_texts = [q + '[SEP]' for q in query_texts]
        input_list = list(zip(query_texts, doc_texts))

        
        # With only_second truncation, sometimes this will fail b/c the first sequence is already longer 
        # than the maximum allowed length so batch_encode_plus will fail with an exception.
        # In many IR applications, a document is much longer than a string, so effectively
        # only_second truncation strategy will be used most of the time.
        res : BatchEncoding = tok.batch_encode_plus(batch_text_or_text_pairs=input_list,
                                   padding='longest',
                                   #truncation='only_second',
                                   truncation='longest_first',
                                   max_length= max_query_len + max_doc_len+1,
                                   return_tensors='pt')
        
        # We need to add EOS tokens to the end of the document
        EOS_token_ids = torch.full_like(res.input_ids[:, :1], self.EOS_TOK_ID)
        input_ids = torch.cat((res.input_ids, EOS_token_ids), dim=1)

        ONES = torch.ones_like(input_ids[:, :1])

        attention_mask = torch.cat((res.attention_mask, ONES), dim=1)

        # token_type_ids may be missing
        return (input_ids, attention_mask)

    def forward(self, input_ids, attention_mask):
        outputs: BaseModelOutputWithPoolingAndCrossAttentions = \
            getattr(self, self.config.model_type).forward(input_ids=input_ids,
                                                  attention_mask=attention_mask)
        
        if not self.use_mean_pool:
            eos_reps = outputs[0][:, -1]
        else:
            eos_reps = mean_pool(outputs[0], attention_mask)

        eos_reps = eos_reps.to(torch.float32)
        out = self.eos(self.dropout(eos_reps))
        return out.squeeze(dim=-1)

