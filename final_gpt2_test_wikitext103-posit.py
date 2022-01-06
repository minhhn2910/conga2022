# -*- coding: utf-8 -*-

# Transformers installation


import numpy as np
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from qtorch.quant import posit_quantize, float_quantize, configurable_table_quantize

device = 'cuda'
model_id = 'gpt2-large'
tokenizer = GPT2TokenizerFast.from_pretrained(model_id)


from nlp import load_dataset
test = load_dataset('wikitext', 'wikitext-103-v1', split='test')
encodings = tokenizer('\n\n'.join(test['text']), return_tensors='pt')


def run(weight_table, act_table ):
    import torch
    import torch.nn as nn
    model = GPT2LMHeadModel.from_pretrained(model_id)

    def linear_weight(input):
        #return configurable_table_quantize(input, torch.tensor(weight_table,dtype = torch.float), scale= 1.0)
        return posit_quantize(input,nsize=6, es=1, scale = 16)
        #return float_quantize(input,exp=4, man=1, rounding="nearest")

    def linear_activation(input):
        #return configurable_table_quantize(input,torch.tensor(act_table, dtype=torch.float), scale= 1.0)
        return posit_quantize(input,nsize=6, es=1, scale = 2)
        #return float_quantize(input,exp=4, man=1, rounding="nearest")

    
    def other_weight(input):
        input = posit_quantize(input, nsize=16, es=1)
        return input

    def other_activation(input):

        input = posit_quantize(input, nsize=16, es=1)
        return input        
        
    def forward_pre_hook_linear(m, input):
        return (linear_activation(input[0]),)

    def forward_pre_hook_other(m, input):
        return (other_activation(input[0]),)   
    

    
    model = model.to(device)
    layer_count = 0
    linear_layer_count = 0
    op_count = 0
    #[17th,141th,145th]
    exclude_list = [16,140,144]

    import transformers.modeling_utils as  modeling_utils
    for name, module in model.named_modules():
      #print (type(module))
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear)or isinstance(module, modeling_utils.Conv1D):
            #print (name)
            if (layer_count not in exclude_list) :
                #if ('lm_head' not in name and 'h.0' not in name):
                module.weight.data = linear_weight(module.weight.data)
                module.register_forward_pre_hook(forward_pre_hook_linear)
            else: 
                print ("exclude ", layer_count)
                module.weight.data = other_weight(module.weight.data)
                module.register_forward_pre_hook(forward_pre_hook_other)

            if (isinstance(module, modeling_utils.Conv1D)):
              #print (module.weight.shape)
                op_count = op_count + module.weight.shape[0] *module.weight.shape[1]
            else:
                op_count = op_count + module.in_features*module.out_features

            layer_count = layer_count + 1
        else:
            if hasattr(module, 'weight' ) and isinstance(module.weight.data,torch.FloatTensor):
                        
                module.weight.data = other_weight(module.weight.data)
                module.register_forward_pre_hook(forward_pre_hook_other)
        
    
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

    
