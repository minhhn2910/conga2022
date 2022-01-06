# -*- coding: utf-8 -*-

# Transformers installation


import numpy as np
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from qtorch.quant import posit_quantize, float_quantize, configurable_table_quantize

device = 'cuda'
model_id = 'gpt2-large'
tokenizer = GPT2TokenizerFast.from_pretrained(model_id)

from datasets import load_dataset

#test = load_dataset("lambada", split='test')
test = load_dataset('wikitext', 'wikitext-103-v1', split='test')
#test = load_dataset("ptb_text_only", split='test')
encodings = tokenizer('\n\n'.join(test['text']), return_tensors='pt')


def run(weight_table, act_table ):
    import torch
    import torch.nn as nn
    model = GPT2LMHeadModel.from_pretrained(model_id)

    model = model.to(device)
    layer_count = 0
    linear_layer_count = 0
    op_count = 0

    #print ("MAC operation count ", op_count)
    print ("Layer count ", layer_count)



    #model = model.to(device)


    import torch
    from tqdm import tqdm
    max_length = model.config.n_positions
    stride = 1024
    #stride = 32

    lls = []
    for i in tqdm(range(0, encodings.input_ids.size(1), stride)):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        trg_len = end_loc - i    # may be different from stride on last loop
        input_ids = encodings.input_ids[:,begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:,:-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            log_likelihood = outputs[0] * trg_len

        lls.append(log_likelihood)

    ppl = torch.exp(torch.stack(lls).sum() / end_loc)
    return ppl.item()



print (run ([],[]))


    
