import json
import pandas as pd
import numpy as np
import warnings
import os
from pathlib import Path
from extract_data import get_game_ids
from scipy.stats import binom

n = 24  # 总试验次数
p = 0.25  # 成功的概率

# 生成二项分布数据
x = np.arange(0, 24)
pmf_values = binom.pmf(x, n, p)

def get_shot_data(shot_data_path, game_id):
    home_shot_path = Path.cwd() / shot_data_path / ("stats_home_shots_" + game_id + ".json")
    away_shot_path = Path.cwd() / shot_data_path / ("stats_away_shots_" + game_id + ".json")

    with open(home_shot_path, 'r') as file:
        home_shot_data = json.load(file)

    with open(away_shot_path, 'r') as file:
        away_shot_data = json.load(file)
    
    home_shot_df = pd.DataFrame(home_shot_data['resultSets'][0]['rowSet'], columns=home_shot_data['resultSets'][0]['headers'])
    away_shot_df = pd.DataFrame(away_shot_data['resultSets'][0]['rowSet'], columns=away_shot_data['resultSets'][0]['headers'])

    home_shot_df['event_id'] = home_shot_df['GAME_EVENT_ID']
    away_shot_df['event_id'] = away_shot_df['GAME_EVENT_ID']

    shot_df = pd.concat([home_shot_df, away_shot_df], ignore_index=True).sort_values(by=['event_id'])

    return shot_df

def get_ball_handler_for_possession(all_df):
    ball_positions = all_df[all_df['player_id'] == -1][['moment_id', 'x_loc', 'y_loc']].rename(columns={'x_loc': 'ball_x', 'y_loc': 'ball_y'})
    all_df = all_df.merge(ball_positions, how='left', on='moment_id', suffixes=('_player', '_ball'))
    all_df['distance'] = np.sqrt((all_df['x_loc'] - all_df['ball_x'])**2 + (all_df['y_loc'] - all_df['ball_y'])**2)

    min_indices = all_df[all_df['OffenseTeamId'] == all_df['team_id']].groupby(['moment_id'])['distance'].idxmin()
    ball_handler_df = all_df.loc[min_indices][['moment_id', 'player_id']].rename(columns={'player_id': 'ball_handler'}) 

    all_df = all_df.merge(ball_handler_df, how='left', on='moment_id')
    all_df_lite = all_df.drop(['team_id', 'player_id', 'x_loc', 'y_loc', 'radius', 'ball_x', 'ball_y', 'distance', 'game_clock_diff'], axis = 1)
    all_df_lite = all_df_lite.drop_duplicates(subset=['moment_id'])

    return all_df_lite

def get_final_df(all_df_lite, shot_df):
    final_df = pd.merge(all_df_lite, shot_df, on='event_id', how='left')

    return final_df

def get_action_for_possession(sample_df, final_df):
    sample_df['action'] = pd.NA
    window_size = 2
    if sample_df[~final_df['SHOT_TYPE'].isna()]['SHOT_TYPE'].unique():
        shoot_min = sample_df[~final_df['SHOT_TYPE'].isna()]['MINUTES_REMAINING'].min()
        shoot_second = sample_df[~final_df['SHOT_TYPE'].isna()]['SECONDS_REMAINING'].min()
        shoot_type = sample_df[~final_df['SHOT_TYPE'].isna()]['SHOT_TYPE'].unique()[0]
        shoot_time = shoot_min * 60 + shoot_second
        sample_df['action'] = np.select(
            [
                (sample_df['game_clock'] <= shoot_time + window_size) & (sample_df['game_clock'] >= shoot_time) & (shoot_type == '2PT Field Goal'),
                (sample_df['game_clock'] <= shoot_time + window_size) & (sample_df['game_clock'] >= shoot_time) & (shoot_type == '3PT Field Goal')
            ],
            [
                'shooting',
                'shooting 3-points'
            ]
        )
    shoot_sample_df = sample_df[sample_df['action'].isin(['shooting', 'shooting 3-points'])]
    reset_sample_df = sample_df[~sample_df['action'].isin(['shooting', 'shooting 3-points'])]
    reset_sample_df = reset_sample_df.reset_index()

    for i in range(len(reset_sample_df)):
        current_time = reset_sample_df.loc[i, 'game_clock']
        window = reset_sample_df[(reset_sample_df['game_clock'] > current_time - window_size) & (reset_sample_df['game_clock'] <= current_time)]

        if len(window['ball_handler'].unique()) == 1:
            reset_sample_df.loc[i, 'action'] = 'dribble'
        else:
            reset_sample_df.loc[i, 'action'] = 'pass'

    sample_df = pd.concat([reset_sample_df, shoot_sample_df], axis=0)
    sample_df = sample_df.sort_values(by=['moment_id'], ascending=True)
    sample_df = sample_df.reset_index(drop=True)
    return sample_df

def process_action_for_game(final_df):
    possession_id_list = final_df['PossessionId'].unique()

    result = []
    for id in possession_id_list:
        sample_df = final_df[final_df['PossessionId'] == id]
        sample_df = get_action_for_possession(sample_df, final_df)
        result.append(sample_df)

    final_action_df = pd.concat(result, axis=0)
    final_action_df = final_action_df.sort_values(by=['moment_id'], ascending=True).reset_index(drop=True)
    final_action_df.drop(['index'], axis=1, inplace=True)

    return final_action_df

def check_shooting_moments(final_action_df):
    # remove frames after shooting
    last_shooting_moments = final_action_df[final_action_df['action'].isin(['shooting', 'shooting 3-points'])].groupby('event_id')['moment_id'].max()
    last_shooting_moments = last_shooting_moments.reset_index().rename(columns={'moment_id': 'last_shooting_moment'})

    final_action_df = pd.merge(final_action_df, last_shooting_moments, on='event_id', how='left')

    final_action_df = final_action_df[final_action_df['moment_id'] <= final_action_df['last_shooting_moment']]
    final_action_df.drop('last_shooting_moment', axis=1, inplace=True)

    return final_action_df

def mark_non_shooting_actions(final_action_df):
    # loc_x, loc_y - nan, not shooting action
    final_action_df.rename(columns={'LOC_X': 'SHOT_LOC_X', 'LOC_Y': 'SHOT_LOC_Y', 'PLAYER_ID': 'SHOT_PLAYER_ID'}, inplace=True)
    final_action_df.loc[~final_action_df['action'].isin(['shooting', 'shooting 3-points']), ['SHOT_LOC_X', 'SHOT_LOC_Y', 'SHOT_PLAYER_ID']] = np.nan

    final_action_df = final_action_df[~((final_action_df['action'] == 'shooting') & (final_action_df['SHOT_LOC_X'].isna()))]

    return final_action_df

def get_final_action_data(game_id, final_df, save_path):
    final_action_df = process_action_for_game(final_df)
    final_action_df = check_shooting_moments(final_action_df)
    final_action_df = mark_non_shooting_actions(final_action_df)

    final_action_df = final_action_df[['PossessionId', 'moment_id', 'OffenseTeamId', 'game_clock', 'shot_clock', 'quarter', 'game_id', 'event_id', 'EventDescription', 'EventTime', 'PossessionStartMarginScore', 'PossessionStartTime_Seconds', 'PossessionEndTime_Seconds',
                                'ball_handler', 'Player1Id', 'SHOT_LOC_X', 'SHOT_LOC_Y', 'action', 'SHOT_PLAYER_ID', 'Points']]
    
    print(final_action_df.shape[0])
    # final_action_df.to_csv(Path(save_path) / f"{game_id}_actions.csv", index=False)
    # print(f"Game {game_id} laebeled with actions has been save to {save_path}/{game_id}_actions.csv")

    return final_action_df

def calculate_reward(row):
    points = row['Points']
    shot_clock = row['shot_clock']
    start_margin_score = row['PossessionStartMarginScore']
    
    if(start_margin_score < 0):
      margin_score = -23 if start_margin_score <= -24 else start_margin_score
      score_reward = pmf_values[-margin_score]*10
    else:
      score_reward = 1

    n_b = 720 - row['game_clock']
    # reward = points + (24.0 - shot_clock) / 24.0 + (n_b / 720)*score_reward
    reward = points + (n_b / 720)*score_reward

    return reward

if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    shot_data_path = "data/response_data/game_details"
    raw_movement_path = Path.cwd() / "data/raw_movement"
    possessions_data_path = Path.cwd() / "data/possessions_data"
    action_data_path = Path.cwd() / "data/actions"
    # if not action_data_path.exists():
    #     action_data_path.mkdir(parents=True, exist_ok=True)
    cle_team_id = 1610612739
        
    files = os.listdir(possessions_data_path)
    game_ids = [file.split("_")[0] for file in files]
    # game_ids = ["0021500002", "0021500011", "0021500021"]
    # game_ids = ["0021500453"]
    fg_pct_df = pd.read_csv(Path.cwd() / "data/player_shot_percentage.csv")
    fg_pct_df = fg_pct_df[['PLAYER_ID', 'GAME_ID', '2pt_shot_percentage', '3pt_shot_percentage']]

    players_df = pd.read_csv(Path.cwd() / "data/players_data.csv")
    players_df = players_df[players_df['team_id']==cle_team_id]
    players_df = players_df[['player_id', 'jersey_number', 'position']]

    for game_id in game_ids:
        posession_df  = pd.read_csv(Path(possessions_data_path) / f"{game_id}_possessions.csv")
        print(posession_df.shape[0])
        all_df_lite = get_ball_handler_for_possession(posession_df)
        shot_df = get_shot_data(shot_data_path, game_id)

        final_df = pd.merge(all_df_lite, shot_df, on='event_id', how='left')
        final_df = final_df[final_df['OffenseTeamId'] == cle_team_id]

        final_action_df = get_final_action_data(game_id, final_df, action_data_path)

        action_df = pd.merge(final_action_df, fg_pct_df, how='left', left_on=['game_id', 'ball_handler'], right_on=['GAME_ID', 'PLAYER_ID'])
        action_df = pd.merge(action_df, fg_pct_df, how='left', left_on=['game_id', 'Player1Id'], right_on=['GAME_ID', 'PLAYER_ID'], suffixes=('_ball_handler', '_player1'))
        action_df = pd.merge(action_df, fg_pct_df, how='left', left_on=['game_id', 'SHOT_PLAYER_ID'], right_on=['GAME_ID', 'PLAYER_ID'], suffixes=('', '_shot_player'))

        action_df['2pt_shot_percentage_ball_handler'] = action_df['2pt_shot_percentage_ball_handler']
        action_df['3pt_shot_percentage_ball_handler'] = action_df['3pt_shot_percentage_ball_handler']
        action_df['2pt_shot_percentage_player1'] = action_df['2pt_shot_percentage_player1']
        action_df['3pt_shot_percentage_player1'] = action_df['3pt_shot_percentage_player1']
        action_df['2pt_shot_percentage_shot_player'] = action_df['2pt_shot_percentage']
        action_df['3pt_shot_percentage_shot_player'] = action_df['3pt_shot_percentage']

        action_df.drop(columns=['PLAYER_ID_ball_handler', 'GAME_ID_ball_handler', 'PLAYER_ID_player1', 'GAME_ID_player1', 'PLAYER_ID',
                        'GAME_ID', '2pt_shot_percentage', '3pt_shot_percentage'], inplace=True)
        
        action_df['Reward'] = action_df.apply(calculate_reward, axis=1)

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

        print(action_df.shape[0])
        action_df.to_csv(Path(action_data_path) / f"{game_id}_actions.csv", index=False)
        print(f"Game {game_id} laebeled with actions has been save to {action_data_path}/{game_id}_actions.csv")