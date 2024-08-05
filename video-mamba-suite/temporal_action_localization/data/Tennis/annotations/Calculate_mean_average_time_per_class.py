"""
This script calculates the total and average duration of specific action labels in the tennis games dataset with 3 classes and prints their values.
"""

import json


from Calculate_labels_train_validation_test import process_games
# Load the JSON file

annotations = ["SERVICE", "EXCHANGE", "OTHER"]
subsets = ['train', 'test']
#input_file_path = 'Tennis_games.json'
input_file_path = 'Processed_Simplified_Tennis_games.json'

with open(input_file_path, 'r') as file:
    data = json.load(file)


label_counts, game_counts = process_games(data, annotations, subsets)
print(label_counts)
Labels_time_training = {"SERVICE" : 0, "EXCHANGE" : 0 , "OTHER" : 0}
Labels_event_count_training = label_counts['train']


Labels_time_validation = {"SERVICE" : 0, "EXCHANGE" : 0 , "OTHER" : 0}
Labels_event_count_validation = label_counts['test']

data = data['database']


for game in data :
    if data[game]['subset'] == 'train':
        for action in data[game]['annotations'] :
            Labels_time_training[action['label']]+= action['segment'][1] - action['segment'][0]
    else :
        for action in data[game]['annotations']:
            Labels_time_validation[action['label']] += action['segment'][1] - action['segment'][0]



Average_Labels_time_training = {}
Average_Labels_time_validation = {}

for Label in Labels_time_training :

    Average_Labels_time_training[Label] = Labels_time_training[Label]/Labels_event_count_training[Label]
    Average_Labels_time_validation[Label] = Labels_time_validation[Label] / Labels_event_count_validation[Label]


print(f'training : \n labels time total : {Labels_time_training} \n  labels time average : {Average_Labels_time_training}')
print(f'validation : \n labels time total : {Labels_time_validation} \n labels time average : {Average_Labels_time_validation}')

