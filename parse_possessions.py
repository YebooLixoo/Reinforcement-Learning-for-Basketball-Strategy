import pandas as pd
import os
from pathlib import Path
from pbpstats.client import Client
from extract_data import get_game_ids

def count_points(possession):
    points = 0
    if possession.next_possession != None:
        if possession.offense_team_id != possession.next_possession.offense_team_id:
            points = -(possession.next_possession.start_score_margin) - possession.start_score_margin
        else:
            points = possession.next_possession.start_score_margin - possession.start_score_margin
    # edge case: last possession
    else:
        # print(possession)
        print("last possession")

    return points

def score_for_possessions(game):
    score = {}
    i = 0
    for possession in game.possessions.items:
        for event in possession.events:
            if event.score.copy():
                # print(event.score.copy().keys())
                if i == 0:
                    score[event.event_num] = {}
                    for k, v in event.score.copy().items():
                        score[event.event_num][k] = v
                    # print(score)
                    last_event_score = event.score.copy()
                    # print(score)
                    i += 1
                else:
                    for k, v in event.score.copy().items():
                        if event.event_num not in score:
                            score[event.event_num] = {}
                        score[event.event_num][k] = v - last_event_score[k]
                    last_event_score = event.score.copy()
    
    return score

def create_possessions_from_events(game):
    # separate offensive rebound as the start of a new possession
    possessions_with_oreb = []
    possession_id = 0
    # score = score_for_possessions(game)

    for possession in game.possessions.items:
        possession_id += 1
        game_id = possession.game_id
        period = possession.period
        possession_start_time = possession.start_time
        possession_end_time = possession.end_time
        possession_team_id = possession.offense_team_id
        score = score_for_possessions(game)
        # point = count_points(possession)
        new_possession = []
        oreb_flag = False
        # print(possession.events)

        for event in possession.events:
            off_event = {
                'GameId': str(game_id),
                'Period': period,
                'PossessionId': possession_id,
                'PossessionStartTime': possession_start_time,
                'PossessionEndTime': possession_end_time,
                'OffenseTeamId': possession_team_id,
                'EventNum': event.event_num,
                'EventType': event.event_type,
                'Points': score[event.event_num][possession_team_id] if event.event_num in score and possession_team_id in score[event.event_num] else 0,
                # 'Points': point,
                'EventActionType': event.event_action_type,
                'EventDescription': event.description,
                'EventTime': event.clock,
                'Player1Id': event.player1_id,
                'PossessionStartMarginScore': possession.start_score_margin
            }

            if event.__class__.__name__ == 'StatsRebound':
                # start a new possession if offensive rebound is found
                # rebound between free throws
                if event.oreb == True and event.is_real_rebound:
                    oreb_flag = True
                    oreb_event_time = event.clock
                    # event time as the end time of this possession
                    for old_event in new_possession:
                        old_event['PossessionEndTime'] = oreb_event_time
                    possessions_with_oreb.append(new_possession)
                    possession_id += 1
                    off_event['PossessionId'] = possession_id
                    off_event['PossessionStartTime'] = oreb_event_time

                    new_possession = [off_event]
                else:
                    if oreb_flag == True:
                        # event_time as the start time of new possession
                        off_event['PossessionStartTime'] = oreb_event_time
                    new_possession.append(off_event)
            else:
                if oreb_flag == True:
                    # event_time as the start time of new possession
                    off_event['PossessionStartTime'] = oreb_event_time
                new_possession.append(off_event)

        if new_possession:
            possessions_with_oreb.append(new_possession)

    print(len(possessions_with_oreb))

    return possessions_with_oreb

def parse_possesssions(game):
    possessions = create_possessions_from_events(game)

    flattened_possessions = [event for possession in possessions for event in possession]
    possession_df = pd.DataFrame(flattened_possessions)

    print(possession_df['Points'].sum())
    point_df = possession_df.groupby(['PossessionId'])['Points'].sum().reset_index()
    possession_df = possession_df.merge(point_df, on=['PossessionId'], how='left')
    possession_df.drop(columns=['Points_x'], axis=1, inplace=True)
    possession_df = possession_df.rename(columns={'Points_y': 'Points'})
    possession_df['event_id'] = possession_df['EventNum']
    possession_df.drop(columns=['EventNum'], axis=1, inplace=True)

    return possession_df

def preprocess_combined_game_data(all_df):
    # remove ~ 40,0000 frames
    event_types_to_remove = [3, 8, 9, 10, 11, 12, 13, 14]
    all_df = all_df[~all_df['EventType'].isin(event_types_to_remove)]

    # remove possessions that is a turnover (offensive foul as turnover)
    # 失误回合shot type不为空保留 - solved
    turnover_df = all_df[all_df['EventType'] == 5]
    possessions_to_remove = turnover_df['PossessionId'].unique()
    all_df = all_df[~all_df['PossessionId'].isin(possessions_to_remove)]

    # keep the frames whose game clock is between possesion_start_time and end_time
    all_df['PossessionStartTime_Seconds'] = [int(time.split(':')[0]) * 60 + int(time.split(':')[1]) for time in all_df['PossessionStartTime']]
    all_df['PossessionEndTime_Seconds'] = [int(time.split(':')[0]) * 60 + int(time.split(':')[1]) for time in all_df['PossessionEndTime']]
    all_df = all_df[(all_df['game_clock'] >= all_df['PossessionEndTime_Seconds']) &
                    (all_df['game_clock'] <= all_df['PossessionStartTime_Seconds'])]
    
    # moment_id - game clock gaurantee the same
    temp_df = all_df.groupby('moment_id')['game_clock'].std().reset_index()
    valid_moment_id = temp_df[temp_df['game_clock'] == 0]['moment_id'].values
    all_df = all_df[all_df['moment_id'].isin(list(valid_moment_id))]

    # keep the frames whose game clock decreases for the first time in each possession
    all_df['game_clock_diff'] = all_df.groupby('PossessionId')['game_clock'].diff()
    # first_decrease_indices = all_df[all_df['game_clock_diff'] <= 0].groupby('PossessionId')

    mask = all_df[all_df['game_clock_diff'] > 1].groupby('PossessionId').nth(0).reset_index()[['PossessionId', 'moment_id']]
    all_df = all_df.merge(mask, on='PossessionId', how='left')
    filtered_df = all_df[(all_df['moment_id_x'] < all_df['moment_id_y']) | (all_df['moment_id_y'].isna())]
    all_df = filtered_df.copy()

    # remove moments that no ball is recorded and only ball is recorded
    moment_counts = all_df['moment_id_x'].value_counts()
    valid_moments = moment_counts[moment_counts == 11].index
    all_df = all_df[all_df['moment_id_x'].isin(valid_moments)]

    all_df = all_df.drop(['moment_id_y'], axis = 1)
    all_df = all_df.rename(columns = {'moment_id_x': 'moment_id'})

    return all_df

def get_final_combined_game_data(game_id, possession_df, movement_df):
    all_df = movement_df.merge(possession_df,  how='inner', on=['event_id'])
    all_df = preprocess_combined_game_data(all_df)
    print(all_df.shape[0])

    save_path = Path.cwd() / "data/possessions_data"
    all_df.to_csv(Path(save_path) / f"{game_id}_possessions.csv", index=False)
    print(f"Game posessions has been save to {save_path}/{game_id}_possessions.csv")

def process_all_games(game_ids, movement_path):
    for game_id in game_ids:
        game = client.Game(game_id)

        possession_df = parse_possesssions(game)
        movement_df = pd.read_csv(Path(movement_path) / f"{game_id}.csv")

        get_final_combined_game_data(game_id, possession_df, movement_df)

if __name__ == "__main__":
    raw_movement_path = Path.cwd() / "data/raw_movement"
    movement_csv_path = Path.cwd() / "data/movement"
    response_data_path = Path.cwd() / "data/response_data"
    possessions_data_path = Path.cwd() / "data/possessions_data"
    if not possessions_data_path.exists():
        possessions_data_path.mkdir(parents=True, exist_ok=True)

    # instantiate the client and instantiate the Game data object for the given game id with possession data.
    settings = {
        "dir": response_data_path,
        "Possessions": {"source": "file", "data_provider": "stats_nba"},
    }
    client = Client(settings)

    game_ids = get_game_ids(raw_movement_path)
    # game_ids = ["0021500002", "0021500011", "0021500021"]
    # game_ids = ["0021500367"]
        
    process_all_games(game_ids, movement_csv_path)
    # game = client.Game(game_ids[0])
    # client = Client(settings)
    # game_id = "0021500367"
    # game = client.Game(game_id)
    # df = create_possessions_from_events(game)