#!/bin/bash
python /home/dazhou/ReasonEval/t-codes/evaluate_temperature_results.py --temperatures 2.0 --models Abel-7B-002 --gpu 1

# python /home/dazhou/ReasonEval/t-codes/evaluate_temperature_results.py --temperatures 1.0 1.3 --models Abel-7B-002 --gpu 4

python /home/dazhou/ReasonEval/t-codes/evaluate_temperature_results.py --temperatures 0.6 1.6 2.0 --models WizardMath-7B-V1.1 --gpu 4