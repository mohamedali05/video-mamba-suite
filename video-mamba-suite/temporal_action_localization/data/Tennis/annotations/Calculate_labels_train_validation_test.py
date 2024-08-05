

"""
This script analyzes tennis game annotation data from various datasets and calculates statistics,
including label counts and percentages for different subsets (e.g., train, test, validation).
It supports all the annotated data present in the 'annotations' folder.

Key functionalities include :
1. **Processing Annotations:**
   - Counts the occurrences of each label in the dataset's annotations.
   - Counts the number of games in each subset (e.g., train, validation, test).
2. **Calculating Statistics:**
   - Computes the total counts of each label across for each subset.
   - Calculates the percentage distribution of each label within all the subsets

### Command-line Arguments:
- `--data_type`: Specifies the type of dataset to analyze. Options include:
  - `'tennis_hayden_set'`: Analyzes the Hayden tennis SET .
  - `'tennis_games'`: Analyzes the general tennis games dataset.
  - `'tennis_3_classes'`: Analyzes the simplified version of 'tennis_games' with three classes .
  - `'test_set'`: Analyzes the test set.

"""
import argparse
import json

def load_json_data(file_path):
    """ Load and return JSON data from a given file path. """
    with open(file_path, 'r') as file:
        return json.load(file)

def process_games(data, annotations, subsets):
    """ Process games data for specified annotations and subsets.
    Args:
        data (dict): The loaded JSON data.
        annotations (list): List of annotation labels to track.
        subsets (list): List of dataset subsets (e.g., train, test)

    Returns:
    tuple: (label_counts, game_counts)
    - label_counts(dict): Counts of each annotation label per subset.
    - game_counts(dict): Counts of games per subset.
    """
    label_counts = {subset: {anno: 0 for anno in annotations} for subset in subsets}
    game_counts = {subset: 0 for subset in subsets}

    data = data['database']

    for game, details in data.items():
     subset = details['subset']
     if subset in subsets:
         for action in details['annotations']:
             label_counts[subset][action['label']] += 1
         game_counts[subset] += 1

    return label_counts, game_counts

def calculate_statitics(label_counts, annotations ,subsets) :
    total_counts = {anno: sum(label_counts[subset][anno] for subset in subsets) for anno in annotations}
    label_percentages = {subset: {anno: 0 for anno in annotations} for subset in subsets}
    for anno in annotations :
     for subset in subsets :
         label_percentages[subset][anno] = round(label_counts[subset][anno] / total_counts[anno], 2)
    return label_percentages



def print_summary(label_counts, game_counts, label_percentages):
    """ Print a summary of label counts and game counts. """
    for subset, counts in label_counts.items():
        print(f"{subset}_labels_count")
        print(counts)

    for subset, percentage in label_percentages.items():
        print(f"{subset}_labels_percentage")
        print(percentage)
    for subset, count in game_counts.items():
        print(f'Count {subset} games: {count}')

def main(args):
    if args.data_type == 'tennis_hayden_set':
        input_file_path = 'Tennis_hayden.json'
        annotations = ["OTH", "SFI", "SFF", "SFL", "SNI", "SNF", "SNL", "HFL", "HFR", "HNL", "HNR"]
        subsets = ['train', 'validation', 'test']
    elif args.data_type == 'tennis_games':
        input_file_path = 'Tennis_games.json'
        annotations = ["OTH", "SFI", "SFF", "SFL", "SNI", "SNF", "SNL", "HFL", "HFR", "HNL", "HNR"]
        subsets = ['train', 'test']
    elif args.data_type == 'tennis_3_classes':
        input_file_path = 'Processed_Simplified_Tennis_games.json'
        annotations = ["SERVICE", "EXCHANGE", "OTHER"]
        subsets = ['train', 'test']
    elif args.data_type == 'test_set' :
        input_file_path = 'All_test_games_ground_truth.json'
        annotations = ["SERVICE", "EXCHANGE", "OTHER"]
        subsets = ['test']


    data = load_json_data(input_file_path)
    label_counts, game_counts = process_games(data, annotations, subsets)
    percentages= calculate_statitics(label_counts, annotations,subsets)
    print_summary(label_counts, game_counts, percentages)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', default='tennis_games')
    args = parser.parse_args()
    main(args)
