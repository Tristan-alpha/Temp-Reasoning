#!/bin/bash
# python /home/dazhou/ReasonEval/t-codes/evaluate_results.py --temperatures 0.1 0.2 0.4 0.5 0.7 0.9 1.1 1.2 1.4 1.5 1.7 1.9 --models Abel-7B-002 --gpu 0 --dataset_name mr-gsm8k

# python /home/dazhou/ReasonEval/t-codes/evaluate_results.py --temperatures 1.0 1.3 --models Abel-7B-002 --gpu 4

python /home/dazhou/ReasonEval/t-codes/evaluate_results.py --temperatures 0.1 0.2 0.4 0.5 0.7 0.9 1.1 1.2 1.4 1.5 1.7 1.9 --models WizardMath-7B-V1.1 --gpu 1 --dataset_name mr-gsm8k