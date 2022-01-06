# -*- coding: utf-8 -*-


import os
import torch
from tqdm import tqdm
import time
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from qtorch.quant import posit_quantize

from transformers import (
    AlbertConfig,
    AlbertForQuestionAnswering,
    AlbertTokenizer,
    squad_convert_examples_to_features
)

from transformers.data.processors.squad import SquadResult, SquadV2Processor, SquadExample

from transformers.data.metrics.squad_metrics import compute_predictions_logits, squad_evaluate

use_own_model = False

if use_own_model:
  model_name_or_path = "/content/model_output"
else:
  model_name_or_path = "ktrapeznikov/albert-xlarge-v2-squad-v2"

output_dir = ""

# Config
n_best_size = 1
max_answer_length = 30
do_lower_case = True
null_score_diff_threshold = 0.0

def to_list(tensor):
    return tensor.detach().cpu().tolist()

# Setup model
config_class, model_class, tokenizer_class = (
    AlbertConfig, AlbertForQuestionAnswering, AlbertTokenizer)
config = config_class.from_pretrained(model_name_or_path)
tokenizer = tokenizer_class.from_pretrained(
    model_name_or_path, do_lower_case=True)
model = model_class.from_pretrained(model_name_or_path, config=config)

import torch.nn as nn

# attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
# attention_probs = self.dropout(attention_probs)
# context_layer = torch.matmul(attention_probs, value_layer)
layer_count = 0
op_count = 0
for name, module in model.named_modules():
  print (name)
  if isinstance(module, nn.GELU) or isinstance(module, nn.Tanh):
    print (module)
  if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
    #print (module)
    layer_count = layer_count + 1
    op_count = op_count + module.in_features*module.out_features
    #print(name, module.in_features)

print ("total dense layer %d \n total MAC_count %d \n" %(layer_count, op_count))   
print ("------------------")

def linear_weight(input):
  return posit_quantize(input,nsize=6, es=1, scale = 32)

def other_weight(input):
  return posit_quantize(input,nsize=8, es=1)  

def linear_activation(input):
  return posit_quantize(input,nsize=6, es=1)

def other_activation(input):
  return posit_quantize(input,nsize=8, es=1)   

def forward_pre_hook_linear(m, input):
    return (linear_activation(input[0]),) 

def forward_hook(m, input,output):
    return other_activation(output)  

def forward_pre_hook_other(m,input):
  if isinstance(input[0], torch.Tensor):
    if (input[0].dtype == torch.float32):
      return (other_activation(input[0]),) 
    else:
      return input
  else:
    return input

layer_count = 0
op_count = 0
#assign hooks to preprocess and post-process layers.
for name, module in model.named_modules():
  if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
    module.weight.data = linear_weight(module.weight.data)
    layer_count = layer_count + 1
    op_count = op_count + module.in_features*module.out_features
    module.register_forward_pre_hook(forward_pre_hook_linear)
    module.register_forward_hook(forward_hook)
  else: #use posit16 for other layers 'weight
    if hasattr(module, 'weight'):
      module.register_forward_pre_hook(forward_pre_hook_other)
      module.weight.data = other_weight(module.weight.data)
      module.register_forward_hook(forward_hook)
    
print ("total processed dense layer %d \n total MAC_count %d \n" %(layer_count, op_count))   
print ("------------------")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "cpu"
model.to(device)

processor = SquadV2Processor()

def run_prediction(question_texts, context_text):
    """Setup function to compute predictions"""
    examples = []

    for i, question_text in enumerate(question_texts):
        example = SquadExample(
            qas_id=str(i),
            question_text=question_text,
            context_text=context_text,
            answer_text=None,
            start_position_character=None,
            title="Predict",
            is_impossible=False,
            answers=None,
        )

        examples.append(example)

    features, dataset = squad_convert_examples_to_features(
        examples=examples,
        tokenizer=tokenizer,
        max_seq_length=384,
        doc_stride=128,
        max_query_length=64,
        is_training=False,
        return_dataset="pt",
        threads=1,
    )

    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=10)

    all_results = []

    for batch in eval_dataloader:
        model.eval()
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }

            example_indices = batch[3]

            outputs = model(**inputs)

            for i, example_index in enumerate(example_indices):
                eval_feature = features[example_index.item()]
                unique_id = int(eval_feature.unique_id)

                output = [to_list(output[i]) for output in outputs]

                start_logits, end_logits = output
                result = SquadResult(unique_id, start_logits, end_logits)
                all_results.append(result)

    output_prediction_file = "predictions.json"
    output_nbest_file = "nbest_predictions.json"
    output_null_log_odds_file = "null_predictions.json"

    predictions = compute_predictions_logits(
        examples,
        features,
        all_results,
        n_best_size,
        max_answer_length,
        do_lower_case,
        output_prediction_file,
        output_nbest_file,
        output_null_log_odds_file,
        False,  # verbose_logging
        True,  # version_2_with_negative
        null_score_diff_threshold,
        tokenizer,
    )

    return predictions


def evaluate( prefix=""):
    #dataset, examples, features = load_and_cache_examples(args, tokenizer, evaluate=True, output_examples=True)

    #if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
    #    os.makedirs(args.output_dir)
    #/content/dataset/dev-v2.0.json
    examples = processor.get_dev_examples("/content/dataset", "dev-v2.0.json")
    examples = examples[:1000]
    print(len(examples))
    features, dataset = squad_convert_examples_to_features(
                examples=examples,
                tokenizer=tokenizer,
                max_seq_length=384,
                doc_stride=128,
                max_query_length=64,
                is_training=False,
                return_dataset="pt",
                threads=1,
            )
    
    #features = features[:1000]
    #dataset = dataset[:1000]
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=32)


    # Eval!
    print("***** Running evaluation {} *****".format(prefix))
    print("  Num examples = %d", len(dataset))
    print("  Batch size = %d", 32)

    all_results = []
    #start_time = timeit.default_timer()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }

            example_indices = batch[3]

            outputs = model(**inputs)

            for i, example_index in enumerate(example_indices):
                eval_feature = features[example_index.item()]
                unique_id = int(eval_feature.unique_id)

                output = [to_list(output[i]) for output in outputs]

                start_logits, end_logits = output
                result = SquadResult(unique_id, start_logits, end_logits)
                all_results.append(result)

            #all_results.append(result)

    # Compute predictions
    #output_prediction_file = os.path.join(args.output_dir, "predictions_{}.json".format(prefix))
    #output_nbest_file = os.path.join(args.output_dir, "nbest_predictions_{}.json".format(prefix))

    #output_null_log_odds_file = os.path.join(args.output_dir, "null_odds_{}.json".format(prefix))

    output_prediction_file = "predictions.json"
    output_nbest_file = "nbest_predictions.json"
    output_null_log_odds_file = "null_predictions.json"


    predictions = compute_predictions_logits(
        examples,
        features,
        all_results,
        n_best_size,
        max_answer_length,
        do_lower_case,
        output_prediction_file,
        output_nbest_file,
        output_null_log_odds_file,
        False,  # verbose_logging
        True,  # version_2_with_negative
        null_score_diff_threshold,
        tokenizer,
    )

    # Compute the F1 and exact scores.
    results = squad_evaluate(examples, predictions)
    return results

"""## 4.0 Run predictions

Now for the fun part... testing out your model on different inputs. Pretty rudimentary example here. But the possibilities are endless with this function.
"""

context = """New Zealand (Māori: Aotearoa) is a sovereign island country in the southwestern Pacific Ocean.
           It has a total land area of 268,000 square kilometres (103,500 sq mi), and a population of 4.9 million.
            New Zealand's capital city is Wellington, and its most populous city is Auckland."""
questions = ["How many people live in New Zealand?", 
             "What's the largest city?"]

# Run method
predictions = run_prediction(questions, context)

# Print results
for key in predictions.keys():
  print(predictions[key])


context = """
            The Battle of Goodenough Island was a battle of the Pacific campaign of World War II fought between 22 and 27 October 1942. Japanese forces had been stranded on Goodenough Island, Papua, during the Battle of Milne Bay. Aircraft and ships headed from Milne Bay to Buna and vice versa had to pass close to Goodenough Island, and a presence on the island could provide warning of enemy operations. The island also had flat areas suitable for the construction of emergency airstrips. The Allies attacked the island prior to the Buna campaign.
          """
questions = ["When was the battle of goodenough island?", 
             "When did the battle of goodenough island end?"]
# Run method
predictions = run_prediction(questions, context)
# Print results
for key in predictions.keys():
  print(predictions[key])



squad_score = evaluate()
print (squad_score)

"""## 5.0 Next Steps

In this tutorial, you learnt how to fine-tune an ALBERT model for the task of question answering, using the SQuAD dataset. Then, you learnt how you can make predictions using the model. 

We retrofitted `compute_predictions_logits` to make the prediction for the purpose of simplicity and minimising dependencies in the tutorial. Take a peak inside that module to see how it works. If you want to serve this as an API, you will want to strip out a lot of the stuff it's doing (such as writing the predictions to a JSON, etc)

You can now turn this into an API by serving it using a web framework. I recommend checking out FastAPI, which is what [Albert Learns to Read](https://littlealbert.now.sh) is built on. 

Feel free to open an issue in the [Github respository](https://github.com/spark-ming/albert-qa-demo/) for this notebook, or tweet me @techno246 if you have any questions! 


"""