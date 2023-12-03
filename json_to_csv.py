import pandas as pd
import numpy as np
import os
import sys

import math
import json
from pathlib import Path
from extract_data import get_game_ids

def json_to_csv(data_path, csv_path):
    files = os.listdir(data_path)
    movement_headers = ["team_id", "player_id", "x_loc", "y_loc", "radius", "game_clock", "shot_clock", "quarter", "game_id", "event_id", "jersey", "abbreviation", "moment_id", "timestamp"]
    count = 0
    for file in files:
        if '.json' not in file:
            continue
        try:
            file_data = open('%s/%s' % (data_path, file))
            game_data = json.load(file_data)

            game_id = game_data['gameid']
            events = game_data['events']

            moments = []
            moment_id = 0

            for event in events:
                event_id = event['eventId']

                visitor_teamid = event['visitor']['teamid']
                visitor_abbreviation = event['visitor']['abbreviation']
                visitor_players = event['visitor']['players']
                home_teamid = event['home']['teamid']
                home_abbreviation = event['home']['abbreviation']
                home_players = event['home']['players']

                movement_data = event['moments']
                # [quarter, miliseconds, game_clock (sec remaining in quarter), shot_clock, None, player_details[]]
                for moment in movement_data:
                    moment_id += 1
                    for location in moment[5]:
                        if location[0] == visitor_teamid:
                            for player in visitor_players:
                                if player['playerid'] == location[1]:
                                    location.extend((moment[2], moment[3], moment[0], game_id, event_id, player['jersey'], visitor_abbreviation,  moment_id, moment[1]))
                        elif location[0] == home_teamid:
                            for player in home_players:
                                if player['playerid'] == location[1]:
                                    location.extend((moment[2], moment[3], moment[0], game_id, event_id, player['jersey'], home_abbreviation,  moment_id, moment[1]))
                        else:
                            location.extend((moment[2], moment[3], moment[0], game_id, event_id, None, None, moment_id, moment[1]))
                        moments.append(location)

            movement = pd.DataFrame(moments, columns=movement_headers)
            movement.to_csv('%s/%s.csv' % (csv_path, game_id), index=False)

            count += 1

            print('\n')
            print('Finished collecting dataframe for Game ID: %s' % game_id)
            print('Completed : %s games.' % count)
        except Exception as e:
            print('Error in loading: %s file, Error: %s' % (str(file), str(e)))

    print('\n')
    print('Finished collecting dataframes for all games.')
    print('Completed : %s games.' % count)

if __name__ == "__main__":
    raw_movement_path = Path.cwd() / "data/raw_movement"
    movement_csv_path = Path.cwd() / "data/movement"

    if not movement_csv_path.exists():
        movement_csv_path.mkdir(parents=True, exist_ok=True)

    json_to_csv(raw_movement_path, movement_csv_path)

