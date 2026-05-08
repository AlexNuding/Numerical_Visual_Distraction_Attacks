# Numerical_Visual_Distraction_Attacks
Code for Master Thesis

# Contents 

## Attack.py
Contains the code for the attacks on Qwen3-VL And llava-v1.6, the specific model and dataset can be selected via the console arguments and other important parameters can be ajusted similarly

## Attack_Claude.py
Contains the code to attack Claude Sonnet 4.6 via the API using a multithreading aproach. Requires local Environmetal variable "ANTHROPIC_API_KEY" to be set to a valid Claude API Key.
Parameters:
### --dataset_path
Path to the dataset that should be used. Example: "./Data/SafetyBench_Main_V2/"
### --output_path
Path where the output .jsonl file should be saved. Example: "./results/Model_claude/safetybench/main_v2.jsonl"
### --tmpfile_location
The programm will save thread results in temporary files taht are deleted upon sucessfull completion. This allows for the specification of where these files will be located. Example: "./results/Model_claude/safetybench/"
### --workers
Number of threads that are to be used. Default value at 6.
### --local
Flag to include if a locally saved dataset is used instead of one from HuggingFace
### --dataset_type
Specification of the used dataset type. To be set to either "SAFETYBENCH" or "HADES" depending on which dataset is used.

## Attack_GPT-5.py
Contains the code to attack GPT-5 via the API using a multithreading aproach. Requires local Environmetal variable "OPENAI_API_KEY" to be set to a valid OpenAI API Key. Contains the exact same parameters for selection as Attack_Claude.py.

## Judge_Harmfulness.py and Judge_Toxicity.py
includes the code for the evalaution of model outputs via GPT-5 as a judge evaluating binary harmfulness and a toxicity score respectively

## Data_Evaluation.py
contains evaluation code for the judgements and to 
