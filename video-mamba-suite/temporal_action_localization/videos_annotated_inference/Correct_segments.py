import json

Anno  = ['PHASE DE NON-JEU' , 'SERVICE', 'ECHANGE']

with open('tennis_games_paolini_andreeva_modified_manually.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

for game in data['database'].items() :
    #print(game)
    fps = game[1]['fps']

    for anno in game[1]['annotations'] :
        anno['segment'] = [round(anno['segment(frames)'][0]/fps,2) , round(anno['segment(frames)'][1]/fps,2)]



with open('tennis_games_paolini_andreeva_modified_manually_2.json', 'w', encoding='utf-8') as output_file:
    json.dump(data, output_file, indent=4)
