import pandas as pd
import os
from pbpstats.client import Client

movement_json_path = os.path.join(os.getcwd(), "data/raw_movement")
pbp_json_path = os.path.join(os.getcwd(), "data/response_data/pbp")
response_data_path = os.path.join(os.getcwd(), "data/response_data")
file = os.listdir(movement_json_path)[0]
game_id = file.split(".")[0]
print(game_id)

settings = {
    "dir": response_data_path,
    "Possessions": {"source": "file", "data_provider": "stats_nba"},
}

client = Client(settings)
game = client.Game(game_id)

for possession in game.possessions.items:
    for event in possession.events:
            print(event)
