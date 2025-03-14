# # Evaluate only Abel-7B-002
# python /home/dazhou/ReasonEval/t-codes/evaluate_temperature_results.py --models Abel-7B-002

# # Evaluate only WizardMath-7B-V1.1
# python /home/dazhou/ReasonEval/t-codes/evaluate_temperature_results.py --models WizardMath-7B-V1.1

# # Evaluate both (default behavior)
# python /home/dazhou/ReasonEval/t-codes/evaluate_temperature_results.py

# # With custom paths
# python /home/dazhou/ReasonEval/t-codes/evaluate_temperature_results.py --models Abel-7B-002 --results_dir /path/to/results --output_dir /path/to/output



# # Run with default settings
# python /home/dazhou/ReasonEval/t-codes/temperature_study_script.py

# # Run only for Abel-7B-002 with custom temperatures
# python /home/dazhou/ReasonEval/t-codes/temperature_study_script.py --models Abel-7B-002 

python /home/dazhou/ReasonEval/t-codes/temperature_study_script.py --gpu 3 --subset_size 500 --temperatures 1.0 1.3 1.6 2.0 --models WizardMath-7B-V1.1

# # Run with a small subset of the dataset for testing
# python /home/dazhou/ReasonEval/t-codes/temperature_study_script.py --subset_size 5