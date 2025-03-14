import json

test_set_filepath = './dataset/mr-gsm8k.json'
test_dataset = []
with open(test_set_filepath) as f:
    for line in f:
        test_dataset.append(json.loads(line))

for key in test_dataset[0].keys():
        print(key)