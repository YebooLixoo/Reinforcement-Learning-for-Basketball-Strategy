import pandas as pd
import numpy as np
import os
import json
from pathlib import Path

def get_players_data(game_data):
    # A dict containing home players data	
    home = game_data["events"][0]["home"]	
    # A dict containig visiting players data	
    visitor = game_data["events"][0]["visitor"]

    # initialize new dictionary	
    all_players = []	

    # Add the values we want for the players (team_id, name, jersey number and position)
    for player in home["players"]:
        all_players.append({
            "player_id": player['playerid'],
                "team_id": home['teamid'],
                "first_name": player['firstname'],
                "last_name": player['lastname'],
                "jersey_number": player['jersey'] if player['jersey'] else 99,
                "position": player['position']
                })
        
    for player in visitor["players"]:	 
        all_players.append({
            "player_id": player['playerid'],
                "team_id": visitor['teamid'],
                "first_name": player['firstname'],
                "last_name": player['lastname'],
                "jersey_number": player['jersey'],
                "position": player['position']
                })

    return all_players

def get_players_dict(game_df):
    # A dict containing home players data	
    home = game_df["events"][0]["home"]	
    # A dict containig visiting players data	
    visitor = game_df["events"][0]["visitor"]
    # creates the players list with the home players	
    players = home["players"]	
    # Then add on the visiting players	
    players.extend(visitor["players"])	

    # initialize new dictionary	
    players_dict = {}	

    # Add the values we want for the players (name and jersey number)
    for player in players:	
        players_dict[player['playerid']] = [player["firstname"]+" "+player["lastname"], player["jersey"]]	

    # Add an entry for the ball
    players_dict.update({-1: ['ball', np.nan]})	

    return players_dict 

if __name__=="__main__":
    raw_moment_path = Path.cwd() / "data/raw_movement"
    raw_movement_files = os.listdir(raw_moment_path)
    
    # all_players = []
    # for file_name in raw_movement_files:
    #     raw_moment_file = Path(raw_moment_path) / file_name
    #     with open(raw_moment_file, 'r') as file:
    #         data = json.load(file)

    #     players_data = get_players_data(data)
    #     print(len(players_data))
    #     all_players.extend(players_data)

    # players_df = pd.DataFrame(all_players)
    # players_df = players_df.drop_duplicates(subset=['player_id', 'team_id'])
    # players_df.to_csv(Path.cwd() / "data/players_data.csv", index=False)

    cle_team_id = 1610612739
    players_df = pd.read_csv(Path.cwd() / "data/players_data.csv")
    players_df = players_df[players_df['team_id']==cle_team_id]
    players_df = players_df[['player_id', 'jersey_number', 'position']]
    print(players_df)

    possessions_data_path = Path.cwd() / "data/possessions_data"
    action_data_path = Path.cwd() / "data/actions"

    files = os.listdir(possessions_data_path)
    game_ids = [file.split("_")[0] for file in files]
    for game_id in game_ids[:1]:
        action_df = pd.read_csv(Path(action_data_path) / f"{game_id}_actions.csv")

        action_df = action_df.merge(players_df, how='left', left_on=['ball_handler'], right_on=['player_id'])
        action_df = action_df.merge(players_df, how='left', left_on=['Player1Id'], right_on=['player_id'], suffixes=('_ball_handler', '_player1'))
        action_df = action_df.merge(players_df, how='left', left_on=['SHOT_PLAYER_ID'], right_on=['player_id'], suffixes=('', '_shot_player'))

        action_df['jersey_number_ball_handler'] = action_df['jersey_number_ball_handler']
        action_df['position_ball_handler'] = action_df['position_ball_handler']
        action_df['jersey_number_player1'] = action_df['jersey_number_player1']
        action_df['position_player1'] = action_df['position_player1']
        action_df['jersey_number_shot_player'] = action_df['jersey_number']
        action_df['position_shot_player'] = action_df['position']

        action_df.drop(columns=['player_id_ball_handler', 'player_id_player1', 'player_id', 'jersey_number', 'position'], inplace=True)
        print(action_df)
        print(action_df.keys())
        print(action_df.shape[0])
        action_df.to_csv(Path(action_data_path) / f"{game_id}_actions.csv", index=False)
        print(f"Game {game_id} laebeled with actions has been save to {action_data_path}/{game_id}_actions.csv")