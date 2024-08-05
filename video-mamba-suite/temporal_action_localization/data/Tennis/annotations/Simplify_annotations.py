
"""
This script processes a JSON file containing tennis game annotations and simplifies the label categories. The key actions performed by the script are:

   - Replacing detailed service-related labels with a unified 'SERVICE' label and assigns it a label_id of 1.
   - Consolidating all other action labels (except 'OTH') into an 'EXCHANGE' label with a label_id of 2.
   - Changing the 'OTH' label to 'OTHER'.
"""


import json

input_file_path = 'Tennis_games.json'
with open(input_file_path, 'r') as file:
    data = json.load(file)

service_labels = {'SNL', 'SNF', 'SNI', 'SFL', 'SFF', 'SFI'}

# Iterate over the database entries
for game_id, game_data in data['database'].items():
    fps  = game_data['fps']
    for annotation in game_data['annotations']:
        annotation['segment'] = [round(annotation['segment(frames)'][0]/ fps , 2 ) ,
                                 round(annotation['segment(frames)'][1] / fps, 2)
                                 ]

        if annotation['label'] in service_labels:
            annotation['label'] = 'SERVICE'
            annotation['label_id'] = 1
        elif annotation['label'] != 'OTH':
            annotation['label'] = 'EXCHANGE'
            annotation['label_id'] = 2
        else :
            annotation['label'] = 'OTHER'


# Save the modified JSON file
with open('Simplified_Tennis_games_round_2.json', 'w') as file:
    json.dump(data, file, indent=4)

print("Labels have been replaced and saved in 'Simplified_Tennis_games_round_2.json'")