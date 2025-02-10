import torch 
import transformers 
from datasets import load_dataset 

import numpy as np 
from typing import List, Literal, Optional, Tuple, Union 
import argparse 
from tqdm import tqdm 
from termcolor import colored 
from tabulate import tabulate 
import os 
import random 

import re 

from openai import OpenAI 
from concurrent.futures import ThreadPoolExecutor 
from threading import Lock 

import sys 
sys.path.append("/home/yangzho6/Open_Physics_of_LLM_Dataversion") 
from simple_names_three import message, messagetwo, messagethree 

naming_match = {
    "bin": "Igsm/templatezoo/", 
    "tri": "Igsm/templatezooo/", 
} 

class MultiTokenEOSCriteria(transformers.StoppingCriteria): 
    # code borrowed from https://github.com/EleutherAI/lm-evaluation-harness 
    """Criteria to stop on the specified multi-token sequence."""

    def __init__(
        self,
        sequence: str,
        tokenizer: transformers.PreTrainedTokenizer,
        initial_decoder_input_length: int,
        batch_size: int,
    ) -> None:
        self.initial_decoder_input_length = initial_decoder_input_length
        self.done_tracker = [False] * batch_size
        self.sequence = sequence 
        self.sequence_ids = tokenizer.encode(sequence, add_special_tokens=False)
        # we look back for 2 more tokens than it takes to encode our stop sequence
        # because tokenizers suck, and a model might generate `['\n', '\n']` but our `sequence` is `['\n\n']`
        # and we don't want to mistakenly not stop a generation because our
        # (string) stop sequence was output in a different tokenization

        # NOTE: there is a minor danger that this will end up looking back 2 tokens into the past, into the inputs to the model,
        # and stopping generation immediately as a result. With only 2 extra tokens of lookback, this risk is minimized
        # Additionally, in lookback_ids_batch we should prevent ever looking back into the inputs as described.
        self.sequence_id_len = len(self.sequence_ids) + 2
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        # For efficiency, we compare the last n tokens where n is the number of tokens in the stop_sequence
        lookback_ids_batch = input_ids[:, self.initial_decoder_input_length :]

        lookback_ids_batch = lookback_ids_batch[:, -self.sequence_id_len :]

        lookback_tokens_batch = self.tokenizer.batch_decode(lookback_ids_batch)

        for i, done in enumerate(self.done_tracker):
            if not done:
                self.done_tracker[i] = self.sequence in lookback_tokens_batch[i] 
        return False not in self.done_tracker

def set_seed(seed):
    # Python's built-in random module
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU.
        
    # CUDA convolution determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

# Usage
set_seed(42)  # You can use any integer value as the seed 

### Parsing the arguments ### 
parser = argparse.ArgumentParser(description = "CommonSense Reasoning with generation and chain-of-thoughts") 
parser.add_argument("--op", type = int, default = 15, help = "Number of operations") 
parser.add_argument("--ip", type = int, default = 20, help = "Number of items per operation") 
parser.add_argument("--force", action = "store_true", help = "Force the generation of the dataset") 
parser.add_argument("--add_fewshot", action = "store_true", help = "Add few-shot learning to the dataset") 
parser.add_argument("--verbose", action = "store_true", help = "Verbose mode") 
parser.add_argument("--limit", type = int, default = None, help = "Limit the number of examples") 
parser.add_argument("--testsuite", type = str, default = "zero_context", help = "Test suite") 
parser.add_argument("--modelname", type = str, default = "meta-llama/Llama-3.1-8B-Instruct", help = "Model name") 
parser.add_argument('--local_rank', type=int, default=-1,
                    help='local rank passed from distributed launcher') 
parser.add_argument('--d', type = int, default = 2, help = "Difficulty level") 
parser.add_argument('--batch_size', type = int, default = 1,
                    help = "batch size for evaluation") 

suffaddition = "The following questions might be easier to solve by equations." 

os.environ['NCCL_TIMEOUT'] = '1800'  # For 2 hours 
os.environ['TORCH_NCCL_BLOCKING_WAIT'] = '1'
print("NCCL_TIMEOUT {}".format(os.environ['NCCL_TIMEOUT'])) 

args = parser.parse_args() 

print(args) 
device = "cuda:{}".format(args.local_rank) 
print(device) 

openai = OpenAI(
    api_key="", # your credentials 
    base_url="", # api provider url 
) 

def boostcommands(filename, message): 
    aggregated_commands = message 
    with open(filename, "r") as file: 
        cotprompt = file.read() 
        cotprompt = cotprompt.replace("\\n", "") 
        cotprompt = cotprompt.replace("\\t", "") 
        message = cotprompt 
        promptlines = [] 
        for line in cotprompt.split("\n"): 
            if len(line) == 0: 
                continue 
            promptlines.append(line) 
        for line in promptlines: 
            indexsolution = line.find("Solution:") 
            if aggregated_commands[-1]["role"] == "user": 
                aggregated_commands[-1]["content"] += " " + line[: indexsolution + 10] 
            else: 
                aggregated_commands.append({"role": "user", "content": line[: indexsolution + 10]}) 
            if len(line) > indexsolution + 10: 
                aggregated_commands.append({"role": "assistant", "content": line[indexsolution + 10:]}) 
    return aggregated_commands 

### Loading the datasets ### 
def get_dataset(
    opmax = 15, 
    limit = None, 
    datafilepath = None, 
    message = None, 
    mode = None, 
    template = None, 
): 
    messages = [] 
    messageone = message 
    messages = [ 
        {"role": "system", "content": "You are a helpful assistant"}, 
        {"role": "user", "content": messageone}, 
    ] 
    
    def addfield(example): 
        messagetwo = messages.copy() 
        inputtext = "Problem: " + example["problem"] + " Question: " + example["question"] + " Solution: " 
        messagetwo.append({"role": "user", "content": inputtext}) 
        example["message"] = messagetwo 
        return example 
    
    dataset = load_dataset(datafilepath, split = "ops_{}".format(opmax)) 
    dataset = dataset.filter(lambda x: x["template"] == template and x["mode"] == mode) 
    
    dataset = dataset.map(addfield) 
    
    if limit is not None: 
        limit = min(limit, len(dataset)) 
        dataset = dataset.select(range(limit)) 
    
    print(colored("Number of examples: {}".format(len(dataset)), "green")) 
    
    return dataset 

class MaxLengthCriteria(transformers.StoppingCriteria): 
    def __init__(self, max_length: int):
        self.max_length = max_length

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return input_ids.shape[-1] >= self.max_length

def stop_sequences_criteria(
    tokenizer: transformers.PreTrainedTokenizer,
    stop_sequences: List[str],
    initial_decoder_input_length: int,
    batch_size: int,
    max_length: int = 256, 
) -> transformers.StoppingCriteriaList:
    outputstoppingcriteria = transformers.StoppingCriteriaList(
        [
            *[
                MultiTokenEOSCriteria(
                    sequence, tokenizer, initial_decoder_input_length, batch_size
                )
                for sequence in stop_sequences
            ],
        ]
    ) 
    
    return outputstoppingcriteria 

def is_integer(s):
    try:
        int(s)
        return True
    except ValueError:
        return False 

def criteriaoutput(generatedtext, inputexample): 
    correctedanswers = 0 
    totalanswers = 0 
    # parsing the answer key 
    for i in range(len(generatedtext)): 
        totalanswers += 1 
        idx_answer_start = inputexample["solution"].find("Answer: ") 
        idx_answer_end = inputexample["solution"].find(".", idx_answer_start) 
        answer_text = inputexample["solution"][idx_answer_start + len("Answer: ") : idx_answer_end] 
        answer_text = int(answer_text.lower()) 
        
        generatedtext[i] = re.sub('.\x08', 'b', generatedtext[i])
        generatedtext[i] = generatedtext[i].lower() 
        if args.verbose and args.local_rank == 0: 
            print(colored(inputexample["solution"], "yellow"), flush = True) 
            print(colored(generatedtext[i], "cyan"), flush = True) 
        
        idx_generated_begin = -1 
        idx_generated_conclude = -1 
        keywords = ["answer: ", "solution: ", "oxed{", "**answer:** ", "**answer: ", "final answer: answer: ", "\nanswer: ", r"\text{answer: } ",  "is ", "answer: "] # updated 
        keywordsend = [".", ".", "}", ".", "**", ".", ".", None, ".", "\n"] 
        cnt = 0 
        
        while not (idx_generated_begin != -1 and idx_generated_conclude != -1) and cnt < len(keywords): 
            if keywords[cnt] in ["oxed{", "is "]: 
                idx_generated_begin = generatedtext[i].rfind(keywords[cnt]) # this relies on the generated is stopped before generated next question plus onwoards by stop 
            else: 
                idx_generated_begin = generatedtext[i].find(keywords[cnt]) 
            if idx_generated_begin != -1: 
                if keywordsend[cnt] is None: 
                    idx_generated_conclude = idx_generated_begin + len(keywords[cnt]) 
                    while generatedtext[0][idx_generated_conclude].isdigit() == True: 
                        idx_generated_conclude += 1 
                else: 
                    idx_generated_conclude = generatedtext[i].find(keywordsend[cnt], idx_generated_begin + len(keywords[cnt])) 
                if idx_generated_conclude == -1: 
                    idx_generated_conclude = len(generatedtext[i]) 
            cnt += 1 
            if not is_integer(generatedtext[i][idx_generated_begin + len(keywords[cnt - 1]) : idx_generated_conclude]): 
                idx_generated_begin = -1 
                idx_generated_conclude = -1 
                continue # if not this line, it will exit the loop 
        
        if idx_generated_begin == -1: 
            if args.local_rank == 0: 
                print(colored("Answer not found", "red"), flush = True) 
            correctedanswers += 0 
            continue 
        else: 
            try: 
                answergenerated_text = int(generatedtext[i][idx_generated_begin + len(keywords[cnt - 1]) : idx_generated_conclude]) 
            except: 
                if args.local_rank == 0: 
                    print(colored("Answer not found", "red"), flush = True) 
                correctedanswers += 0 
                continue 
            if args.local_rank == 0: 
                if answergenerated_text == answer_text: 
                    print(colored("Answer {} expected {}".format(answergenerated_text, answer_text), "green"), flush = True) 
                else: 
                    print(colored("Answer {} expected {}".format(answergenerated_text, answer_text), "red"), flush = True) 
            correctedanswers += int(answergenerated_text == answer_text) 
    return correctedanswers, totalanswers 

def generateoutput(batch): 
    outputs = openai.chat.completions.create(
        model = "", # input your model name 
        messages = batch["message"], 
        temperature = 0.0, 
        max_tokens = 4096, # standard default, but adjust if you think needed 
    ) 
    generatedtext = outputs.choices[0].message.content 
    generatedtext = [generatedtext] 

    corrected, total = criteriaoutput(generatedtext, batch) 
    return corrected, total 

countaccum = {} 
listtasks = [] 

lock = Lock() 

for mode in ["normalforward", "forwardreverse"]: 
    for template in ["crazy_zootopia", "teachers_in_school", "movie_festival_awards"]: 
        if args.op > 30 and template in ["teachers_in_school", "movie_festival_awards"]: 
            continue 
        task = "op{}_ip{}_force{}_{}_{}".format(args.op, args.ip, args.force, template, mode) 
        listtasks.append(task) 
        countaccum[task] = [0, 0, 0] 
        
        prompttttt = {
            "crazy_zootopia": message, 
            "teachers_in_school": messagetwo, 
            "movie_festival_awards": messagethree, 
        } 
        
        if mode == "forwardreverse": 
            for key in prompttttt.keys(): 
                prompttttt[key] += " " + suffaddition 
        
        limit = None 
        if template == "crazy_zootopia": 
            limit = int(args.limit * 0.8) 
        else: 
            limit = int(args.limit * 0.1) 
        
        # making sure that the reverse mode counts fair 
        promptfilepath = "Igsm/zero_context_2/{}/{}/igsm_op{}_ip20_force_True_{}_{}_cot.txt".format("medium" if args.d == 2 else "hard", args.op, args.op, mode, template) 
        datasetpath = "YangZhoumill/factorreit_{}_{}".format(args.testsuite if args.testsuite != "zero_context" else "zerocontext", "medium" if args.d == 2 else "hard") 

        dataloader = get_dataset(
            adding_fewshot = args.add_fewshot, 
            opmax = args.op, 
            ipmax = args.ip, 
            force = args.force, 
            batch_size = args.batch_size, 
            limit = limit, 
            promptfilepath = promptfilepath, 
            datafilepath = datasetpath, 
            message = prompttttt[template], 
            mode = mode, 
            template = template, 
        ) 

        totalexamples = 0 
        correctanswers = 0 
        batchnum = 160 
        
        progress_bar = tqdm(total=len(dataloader), desc="Processing batches") 

        for i in tqdm(range(0, len(dataloader), batchnum)): 
            with ThreadPoolExecutor(max_workers=batchnum) as executor: 
                futures = [] 
                for j in range(batchnum): 
                    if i + j >= len(dataloader): 
                        break 
                    batch = dataloader[i + j] 
                    future = executor.submit(generateoutput, batch) 
                    futures.append(future) 
                    
                for future in futures:
                    try:
                        corrected, total = future.result() 
                        with lock:  # Safely update shared variables
                            totalexamples += total 
                            correctanswers += corrected 
                            progress_bar.update(1) 
                            print("Corrected {} Total {} Task {}".format(correctanswers, totalexamples, task)) 
                    except Exception as e:
                        print(f"Error in thread: {e}")
        progress_bar.close() 

        # statistics 
        headers = ["Task"] 
        data = [task] 
        
        # Print table 
        print("Here are the statistics for inference") 
        if args.local_rank == 0: 
            print("task {} totalexamples {} correctanswers {}".format(task, totalexamples, correctanswers)) 
            data = [task, totalexamples, correctanswers, correctanswers/totalexamples] 
            print(tabulate([data], headers=headers, tablefmt="grid")) 
            countaccum[task] = [totalexamples, correctanswers, correctanswers / totalexamples] 

if args.local_rank == 0: 
    # formatting the output 
    print(args) 
    
    for mode in ["normalforward", "forwardreverse"]: 
        taskname = "op{}_aggregate_{}".format(args.op, mode) 
        aggregatetotalexamples = 0 
        aggregatecorrectanswers = 0 
        for keys in countaccum.keys(): 
            if mode in keys: 
                aggregatetotalexamples += countaccum[keys][0] 
                aggregatecorrectanswers += countaccum[keys][1] 
        if aggregatetotalexamples == 0: 
            continue 
        countaccum[taskname] = [aggregatetotalexamples, aggregatecorrectanswers, aggregatecorrectanswers / aggregatetotalexamples] 
    
    taskname = "op{}_aggregate".format(args.op) 
    aggregatetotalexamples = 0 
    aggregatecorrectanswers = 0 
    for taske in countaccum.keys(): 
        aggregatetotalexamples += countaccum[taske][0] 
        aggregatecorrectanswers += countaccum[taske][1] 
    countaccum[taskname] = [aggregatetotalexamples, aggregatecorrectanswers, aggregatecorrectanswers / aggregatetotalexamples] 
    
    headers = ["Task", "Total", "Correct", "Solve Rate"] 
    data = [] 
    for task in countaccum.keys(): 
        data.append([task, countaccum[task][0], countaccum[task][1], countaccum[task][2]]) 
    print(tabulate(data, headers = headers, tablefmt = "grid")) 
