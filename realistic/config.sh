#!/bin/bash

# Configure your API Keys and URLs
# You can use environment variables here as well
backend_type='openai' # can be 'openai', 'gemini' and 'anthropic'
export OPENAI_BASE_URL='http://127.0.0.1:30000/v1'
export OPENAI_API_KEY='EXAMPLE API KEY'
export GEMINI_API_KEY=''
export ANTHROPIC_API_KEY=''

# Model and Dataset Configuration
model_name='Qwen/Qwen2.5-7B-Instruct' # API model name
dataset_base='InfiniAILab/gsm_infinite' # Base name for the dataset
save_name='qwen-2.5-7b-instruct' # Model name for saving the results

# Sampling Settings
num_samples=1
temperature=0.0
max_tokens=4096

# Batch size and example limit per op
batch_size=200
limit=200

# Lengths to process (can be numbers or strings like '8k')
lengths=( "0" "8k" "16k" "32k" )

# Dataset suffixes
dataset_suffixes=( "medium" "hard" )

# Operation Range Configuration (Per length and suffix). if empty, the subset will be skipped.
declare -A ops_config
# Example configurations:
ops_config["0_medium"]='{"start": 2, "end": 30, "stride": 1}' 
ops_config["0_hard"]='{"start": 2, "end": 30, "stride": 1}' 
ops_config["8k_medium"]='{"start": 2, "end": 30, "stride": 1}' 
ops_config["8k_hard"]='{"start": 2, "end": 30, "stride": 1}'  
ops_config["16k_medium"]='{"start": 2, "end": 30, "stride": 1}' 
ops_config["16k_hard"]='{"start": 2, "end": 30, "stride": 1}'  
ops_config["32k_medium"]='{"start": 2, "end": 30, "stride": 1}'
ops_config["32k_hard"]='{"start": 2, "end": 30, "stride": 1}'  

# Filter Configuration (JSON string)
filter_config='[
    {"percentage": 0.4, "template": "crazy_zootopia", "mode": "normalforward"},
    {"percentage": 0.05, "template": "movie_festival_awards", "mode": "normalforward"},
    {"percentage": 0.05, "template": "teachers_in_school", "mode": "normalforward"},
    {"percentage": 0.4, "template": "crazy_zootopia", "mode": "forwardreverse"},
    {"percentage": 0.05, "template": "movie_festival_awards", "mode": "forwardreverse"},
    {"percentage": 0.05, "template": "teachers_in_school", "mode": "forwardreverse"}
]'

