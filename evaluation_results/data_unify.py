import pandas as pd
import os

path = "/home/dazhou/ReasonEval/evaluation_results/Abel-7B-002/mr-gsm8k"
original_detailed_path = os.path.join(path, 'original_detailed.csv')
detailed_path = os.path.join(path, 'detailed.csv')
aggregated_path = os.path.join(path, 'aggregated.csv')

# For detailed.csv
data_original_detailed = pd.read_csv(original_detailed_path)
data_detailed = pd.read_csv(detailed_path)
data_detailed['Dataset'] = 'mr-gsm8k'  # Add dataset column
detailed_df = pd.concat([data_original_detailed, data_detailed])
detailed_df.to_csv(detailed_path, index=False)

# # For aggregated.csv
# if os.path.exists(aggregated_path):
#     original_aggregated_path = os.path.join(path, 'original_aggregated.csv')
#     if os.path.exists(original_aggregated_path):
#         data_original_aggregated = pd.read_csv(original_aggregated_path)
#         data_aggregated = pd.read_csv(aggregated_path)
#         data_aggregated['Dataset'] = 'mr-gsm8k'  # Add dataset column
#         aggregated_df = pd.concat([data_original_aggregated, data_aggregated])
#         aggregated_df.to_csv(aggregated_path, index=False)