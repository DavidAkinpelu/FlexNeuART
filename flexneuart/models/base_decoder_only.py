#!/usr/bin/env python
#
#  Copyright 2014+ Carnegie Mellon University
#
#  Using some bits from CEDR (for padding and masking):
#  https://github.com/Georgetown-IR-Lab/cedr
#  which has MIT, i.e., Apache 2 compatible license.
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
from typing import List, Optional

import torch

from flexneuart.models.base import BaseModel
from flexneuart.models.utils import init_lora_model



class DecoderBaseRanker(BaseModel):
    """
       The base class for all Transformer-based ranking models.

       We generally/broadly consider these models to be decoder only-variants, hence, the name of the base class.
    """

    def __init__(self, base_model_flavor, quant_type, lora_target_modules, lora_r, lora_alpha, lora_dropout, load_in_4bit,
                 load_in_8bit, trust_remote_code, use_dora):
        """Decoder-based model ranker constructor.

            :param bart_flavor:   The name of the underlying Transformer or a path
                                  to a previously stored model. This will will be passed
                                  to AutoModel.from_pretrained().
                                  One can use quite a few Transformer models as long as
                                  they return an object of the type:
                                  BaseModelOutputWithPoolingAndCrossAttentions.
        """
        super().__init__()
        init_lora_model(self, base_model_flavor=base_model_flavor, quant_type=quant_type, lora_target_modules=lora_target_modules, 
                        lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, load_in_4bit=load_in_4bit, 
                        load_in_8bit=load_in_8bit, trust_remote_code=trust_remote_code, use_dora=use_dora)
        
        self.quant_type = quant_type
        self.use_dora = use_dora


    def bert_param_names(self):
        """
        :return: a list of the lora-parameters.
        For convention, the funtion names still remain bert_param_names, although
        """
        return set([k for k in self.state_dict().keys() if "lora" in k])
    
    def forward(self, **inputs):
        raise NotImplementedError

    def _tokenize_and_encode(self, text):
        """Tokenizes the text and converts tokens to respective IDs

        :param text:  input text
        :return:      an array of token IDs
        """
        toks = self.tokenizer.tokenize(text)
        return self.tokenizer.convert_tokens_to_ids(toks)
    