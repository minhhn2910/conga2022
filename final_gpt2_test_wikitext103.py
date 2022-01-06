# -*- coding: utf-8 -*-

# Transformers installation


import numpy as np
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from qtorch.quant import posit_quantize, float_quantize, configurable_table_quantize

device = 'cuda'
model_id = 'gpt2-large'
tokenizer = GPT2TokenizerFast.from_pretrained(model_id)

original_table =  np.array([1.0/65536, 1.0/32768, 1.0/16384, 1.0/8192, 1.0/4096, 1.0/2048, 1.0/1024, 1.0/512, 1.0/256, 1.0/128,
               3.0/256, 1.0/64,  5.0/256 , 3.0/128,  7.0/256, 1.0/32, 9.0/256, 5.0/128, 3.0/64, 7.0/128,
               1.0/16,  9.0/128, 5.0/64, 3.0/32,    7.0/64,    1.0/8, 9.0/64, 3.0/16, 1.0/4, 3.0/8, 1.0/2, 1.0])

original_table = original_table*256 #posit(6,1)/4
original_table = np.array([0.0009765625, 0.00390625, 0.0078125, 0.015625, 0.0234375, 0.03125, 
                            0.046875, 0.0625, 0.078125, 0.09375, 0.109375, 0.125, 0.15625, 
                            0.1875, 0.21875, 0.25, 0.28125, 0.3125, 0.34375, 0.375, 0.40625, 
                            0.4375, 0.46875, 0.5, 0.5625, 0.625, 0.6875, 0.75, 0.8125, 0.875,
                            0.9375, 1., 1.125, 1.25, 1.375, 1.5, 1.625, 1.75, 1.875, 2., 2.25, 
                            2.5, 2.75, 3., 3.25, 3.5, 3.75, 4., 5., 6., 7., 8., 10., 12., 14., 
                            16., 24., 32., 48., 64., 128., 256., 1024.])

original_table = np.array([0.003906, 0.015625, 0.031250, 0.062500, 0.093750, 0.125000, 0.187500, 0.250000, 0.312500, 0.375000, 0.437500, 0.500000, 0.625000, 0.750000, 0.875000, 1.000000, 1.250000, 1.500000, 1.750000, 2.000000, 2.500000, 3.000000, 3.500000, 4.000000, 6.000000, 8.000000, 12.000000, 16.000000, 32.000000, 64.000000, 256.000000])

from nlp import load_dataset
#test = load_dataset('wikitext', 'wikitext-103-v1', split='test')
test = load_dataset('wikitext', 'wikitext-103-v1', split='test')
encodings = tokenizer('\n\n'.join(test['text']), return_tensors='pt')


def run(weight_table, act_table ):
    import torch
    import torch.nn as nn
    model = GPT2LMHeadModel.from_pretrained(model_id)

    def linear_weight(input):
        return configurable_table_quantize(input, torch.tensor(weight_table,dtype = torch.float), scale= 1.0)
        #return posit_quantize(input,nsize=6, es=1, scale = 1)
        #return float_quantize(input,exp=4, man=1, rounding="nearest")

    def linear_activation(input):
        return configurable_table_quantize(input,torch.tensor(act_table, dtype=torch.float), scale= 1.0)
        #return posit_quantize(input,nsize=6, es=1, scale = 1)
        #return float_quantize(input,exp=4, man=1, rounding="nearest")

    def forward_pre_hook_linear(m, input):
        return (linear_activation(input[0]),)

    model = model.to(device)
    layer_count = 0
    linear_layer_count = 0
    op_count = 0
    #[17th,141th,145th]
    #exclude_list = [16,140,144]
    exclude_list = []
    import transformers.modeling_utils as  modeling_utils
    for name, module in model.named_modules():
      #print (type(module))
      if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear)or isinstance(module, modeling_utils.Conv1D):
        #print (name)
        if (layer_count not in exclude_list) :
            #if ('lm_head' not in name and 'h.0' not in name):
            module.weight.data = linear_weight(module.weight.data)
            module.register_forward_pre_hook(forward_pre_hook_linear)
        #else: 
            #print ("exclude ", layer_count)

        if (isinstance(module, modeling_utils.Conv1D)):
          #print (module.weight.shape)
          op_count = op_count + module.weight.shape[0] *module.weight.shape[1]
        else:
          op_count = op_count + module.in_features*module.out_features
        
        layer_count = layer_count + 1
    
    #print ("MAC operation count ", op_count)
    print ("Layer count ", layer_count)



    #model = model.to(device)


    import torch
    from tqdm import tqdm
    max_length = model.config.n_positions
    stride = 1024

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

weight_table = np.array([0.01171875, 0.015625 ,  0.0234375 , 0.0390625 , 0.046875 ,  0.0546875,
                 0.0625,     0.078125 ,  0.09375 ,   0.109375,  0.125 ,     0.1875,
                 0.21875  ,  0.375  ,    0.75   ,    1.5     ])
act_table = np.array([ 0.03125 , 0.0625 ,  0.125  ,  0.1875 ,  0.25,     0.625   , 0.75,     0.875,
                      1.   ,    1.25  ,   1.5   ,   2.5  ,    4.   ,    6. ,      8. ,     12.     ])
weight_remove_8 = [15, 14, 1, 13, 9, 5, 12, 4, 6, 8, 7, 11, 10, 3, 2, 0]
act_remove_8 = [7, 8, 6, 1, 9, 2, 10, 14, 3, 15, 13, 4, 5, 11, 12, 0]
weight_table = np.delete(weight_table,weight_remove_8[:8])
#act_table = np.delete(act_table,act_remove_8[:8])


full_table = np.append(weight_table, act_table)
#after search
# full_table = np.array([7.32421875e-03,2.11486816e-02,3.90625000e-02,5.67626953e-02
#                         ,7.32421875e-02,9.66796875e-02,1.56250000e-01,3.73535156e-01
#                         ,7.81250000e-03,5.85937500e-02,1.25000000e-01,1.64062500e-01
#                         ,3.12500000e-01,6.05468750e-01,7.73437500e-01,4.37500000e-01
#                         ,1.00000000e+00,2.10937500e+00,1.59521484e+00,1.25000000e+00
#                         ,4.50000000e+00,6.00000000e+00,8.25000000e+00,3.09375000e+00])

full_table = np.array([6.59179688e-03, 2.19726562e-02, 4.02832031e-02, 6.25000000e-02,
                         8.51440430e-02, 2.10937500e-01, 1.21093750e-01, 5.36407471e-01,
                         1.56250000e-02, 6.25000000e-02, 1.25000000e-01, 1.64062500e-01,
                         2.92968750e-01, 6.25000000e-01, 8.43750000e-01, 4.37500000e-01,
                         1.00000000e+00, 2.11486816e+00, 1.50000000e+00, 1.25000000e+00,
                         4.25000000e+00, 3.08789062e+00, 8.25000000e+00, 6.00000000e+00]
                        )
weight_table = full_table[:8]
act_table = full_table[8:]


print (len(weight_table))
print (weight_table)
print (act_table)
last_know_acc = -run ( weight_table ,act_table )
last_know_acc_log  = [last_know_acc]
print (last_know_acc)
exit()
log_idx = []
current_learning_rate = 0.5
stop = False
while (not stop):
    increase_list = []
    decrease_list = []

    print ('curr table')
    print (full_table)
    print ('curr rate ', current_learning_rate)
    print ('curr acc ', last_know_acc)
    np.savetxt("gpt2_search_reduced.txt", full_table)
    for i in range(len(full_table)):
        #print (i)
        temp_table = np.copy(full_table)
        temp_val = temp_table[i]
        temp_table[i] = temp_val + current_learning_rate * temp_val

        #combined_table_split = np.hsplit(temp_table, 4)
        weight_table = temp_table[:8]
        act_table = temp_table[8:]

        curr_ssim = run ( weight_table ,act_table )
        #all_ssim.append(curr_ssim)
        increase_list.append(-curr_ssim)
        temp_table[i] = temp_val - current_learning_rate * temp_val

        weight_table = temp_table[:8]
        act_table = temp_table[8:]


        curr_ssim = run ( weight_table ,act_table )

        decrease_list.append(-curr_ssim)

    max_dec = max(decrease_list)
    max_inc = max(increase_list)
    if (last_know_acc >= max(max_dec,max_inc) ):
        #stop = True
        current_learning_rate = current_learning_rate/2.0
        if (current_learning_rate < 0.01):
            stop = True
            print ("learning rate too small < 1% . Stop algo")
        continue #continue without modifying the arr
    if (max_dec > max_inc):
        idx = np.argmax(np.array(decrease_list))
        log_idx.append(idx)
        full_table[idx] = full_table[idx] - current_learning_rate * full_table[idx]
    else:
        idx = np.argmax(np.array(increase_list))
        log_idx.append(idx)
        full_table[idx] = full_table[idx] + current_learning_rate * full_table[idx]
    print ("log ")
    print (full_table)
    print (log_idx)

    #stop = True

    last_know_acc = max(max_dec,max_inc)
    last_know_acc_log.append([current_learning_rate,last_know_acc])
    print (last_know_acc_log)

    
