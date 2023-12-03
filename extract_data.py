import pandas as pd
import os
import py7zr
import shutil

def extract_movement_json(zip_dir, output_dir):
    full_zip_dir = os.path.normpath(os.path.join(os.getcwd(), zip_dir))
    files = os.listdir(full_zip_dir)
    cle_files = [file for file in files if "CLE" in file and file.endswith('.7z')]
    # print(cle_files)
    # print(len(cle_files))

    for file in cle_files:
        with py7zr.SevenZipFile(os.path.join(full_zip_dir, file), mode='r') as z:
            z_files = z.getnames()
            json_files = [f for f in z_files if f.endswith('.json')]

            for json_file in json_files:
                if json_file.endswith('.json'):
                    z.extract(path=output_dir, targets=[json_file])
            
            print(f"Extracted JSON files from {file} to {output_dir}")

def get_game_ids(json_dir):
    full_json_dir = os.path.normpath(os.path.join(os.getcwd(), json_dir))
    game_ids = []

    for file_name in os.listdir(full_json_dir):
        file_path = os.path.join(full_json_dir, file_name)
        if os.path.isfile(file_path) and file_name.endswith('.json'):
            game_ids.append(file_name.split('.')[0])
    
    return game_ids

def extract_pbp_json(pbp_dir, output_dir):
    full_out_dir = os.path.normpath(os.path.join(os.getcwd(), output_dir))
    if not os.path.exists(full_out_dir):
        os.makedirs(full_out_dir)
    game_ids = get_game_ids(raw_movement_dir)

    for game_id in game_ids:
        pbp_file_name = "stats_" + game_id + ".json"
        source_dir = os.path.normpath(os.path.join(os.getcwd(), pbp_dir, pbp_file_name))

        if os.path.exists(source_dir):
            shutil.copy2(source_dir, output_dir)
            print(f"File '{pbp_file_name}' copied to '{full_out_dir}'")
        else:
            print(f"File '{pbp_file_name}' not found in '{source_dir}'")

def extract_game_details_json(game_details_dir, output_dir):
    full_out_dir = os.path.normpath(os.path.join(os.getcwd(), output_dir))
    if not os.path.exists(full_out_dir):
        os.makedirs(full_out_dir)
    game_ids = get_game_ids(raw_movement_dir)

    for game_id in game_ids:
        print(game_id)
        away_shots_file_name = "stats_away_shots_" + game_id + ".json"
        home_shots_file_name = "stats_home_shots_" + game_id + ".json"
        away_shots_source_dir = os.path.normpath(os.path.join(os.getcwd(), game_details_dir, away_shots_file_name))
        home_shots_source_dir = os.path.normpath(os.path.join(os.getcwd(), game_details_dir, home_shots_file_name))

        if os.path.exists(away_shots_source_dir):
            shutil.copy2(away_shots_source_dir, output_dir)
            print(f"File '{away_shots_file_name}' copied to '{full_out_dir}'")
        else:
            print(f"File '{away_shots_file_name}' not found in '{away_shots_source_dir}'")

        if os.path.exists(home_shots_source_dir):
            shutil.copy2(home_shots_source_dir, output_dir)
            print(f"File '{home_shots_file_name}' copied to '{full_out_dir}'")
        else:
            print(f"File '{home_shots_file_name}' not found in '{home_shots_source_dir}'")

if __name__ == "__main__":
    zip_dir = "../nba-movement-data-master/data/"
    raw_movement_dir = os.path.normpath(os.path.join(os.getcwd(), "data/raw_movement"))
    if not os.path.exists(raw_movement_dir):
        os.makedirs(raw_movement_dir)

    extract_movement_json(zip_dir, raw_movement_dir)
    
    # raw_pbp_dir = "../data/response_data/pbp"
    # output_pbp_dir = "data/response_data/pbp"
    # extract_pbp_json(raw_pbp_dir, output_pbp_dir)

    # game_details_dir = "../data/response_data/game_details"
    # output_game_details_dir = "data/response_data/game_details"
    # extract_game_details_json(game_details_dir, output_game_details_dir)
