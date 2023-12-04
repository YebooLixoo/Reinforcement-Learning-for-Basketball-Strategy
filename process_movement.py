import pandas as pd
import os
from pathlib import Path

if __name__ == "__main__":
    action_data_path = Path.cwd() / "data/actions"
    movement_csv_path = Path.cwd() / "data/movement"
    processed_movement_path = Path.cwd() / "data/processed_movement"
    action_files = os.listdir(action_data_path)
    cle_team_id = 1610612739
    
    if not processed_movement_path.exists():
        processed_movement_path.mkdir(parents=True, exist_ok=True)

    game_ids = [action_file.split("_")[0] for action_file in action_files]

    for game_id in game_ids:
        action_file = (action_data_path) / f"{game_id}_actions.csv"
        action_df = pd.read_csv(Path(action_data_path) / action_file)

        movement_file = (movement_csv_path) / f"{game_id}.csv"
        movement_df = pd.read_csv(Path(movement_csv_path) / movement_file)

        moment_ids = action_df['moment_id'].tolist()
        filtered_movement_df = movement_df[(movement_df['moment_id'].isin(moment_ids))]

        filtered_movement_df.to_csv(Path(processed_movement_path) / f"{game_id}_processed.csv", index=False)
        print(f"Game {game_id} laebeled with actions has been save to {processed_movement_path}/{game_id}_processed.csv")
        print(filtered_movement_df.shape[0])
