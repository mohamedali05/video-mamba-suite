"""
This script processes the annotations in a JSON file containing tennis game data. The main goals of the script are to refine the labels in the annotations and merge consecutive annotations of the same type. The script includes the following key functionalities:

1. **Processing Annotations:**
   - **Label Refinement:** Changes the 'OTHER' label to 'EXCHANGE' in specific scenarios where it appears between 'EXCHANGE' or 'SERVICE' and 'EXCHANGE' annotations.
   - **Merging Annotations:** Merges consecutive annotations of the same label ('EXCHANGE' or 'SERVICE') into a single annotation, updating the segment times accordingly.
2. **Saving Processed Data:** Writes the modified annotations back to a new JSON file.
"""
import json
def process_annotations(annotations):
    processed = []
    i = 0
    previous = None
    while i < len(annotations):
        current = annotations[i]

        if i > 0:
            previous = annotations[i - 1]

        if i < len(annotations) - 1:
            next_annotation = annotations[i + 1]
        else:
            next_annotation = None

        # Check if current annotation is 'OTHER'
        if current['label'] == 'OTHER' and i>0:
            if (previous['label'] == 'EXCHANGE' and next_annotation and next_annotation['label'] == 'EXCHANGE') or \
                    (previous['label'] == 'SERVICE' and next_annotation and next_annotation['label'] == 'EXCHANGE') :
                current['label'] = 'EXCHANGE'

        # Merge consecutive 'EXCHANGE' or 'SERVICE' annotations
        if i > 0 and (current['label'] == previous['label'])  :
            processed[-1]['segment'][1] = current['segment'][1]
            processed[-1]['segment(frames)'][1] = current['segment(frames)'][1]

        else:
            processed.append(current)

        i += 1

    return processed


input_file_path = 'Simplified_Tennis_games.json'
with open(input_file_path, 'r') as file:
    data = json.load(file)


for game_key in data['database']:
    annotations = data['database'][game_key]['annotations']
    data['database'][game_key]['annotations'] = process_annotations(annotations)


output_file_path = 'Processed_Simplified_Tennis_games.json'
with open(output_file_path, 'w') as f:
    json.dump(data, f, indent=4)

