from collections import deque
import torch
import pandas as pd
from PIL import Image
import numpy as np
import random
import torch.nn.functional as F
from model import DQN
import torch.optim as optim
import matplotlib.pyplot as plt
import datetime

import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO) 
now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
fileinfo = logging.FileHandler(f"./train_log_{now}.log")
fileinfo.setLevel(logging.INFO) 
formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
fileinfo.setFormatter(formatter)
logger.addHandler(fileinfo)

#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
q_net = DQN(3, 4).to(device)
target_q_net = DQN(3, 4).to(device)
gamma = 1
optimizer = optim.Adam(q_net.parameters(), lr=0.0000625, eps=1.5e-4)
global_step = 0
target_update_step = 2048
loss_plot_step = 256
loss_values = []
running_loss = 0
sequence_num = 50
batch_size = 32

action_dict = {
    'dribble': 0,
    'pass': 1,
    'shooting': 2,
    'shooting 3-points': 3
}

action_index_dict = {
    0: 'dribble',
    1: 'pass',
    2: 'shooting',
    3: 'shooting 3-points'
}

def show(y, scale, des, ydes, xdes, path):
    x = [i*scale for i in range(len(y))]
    plt.plot(x, y, 'b-', label=des)
    plt.xlabel(xdes)
    plt.ylabel(ydes)
    plt.legend()
    plt.savefig(path)
    plt.close("all")

train_game_ids = [
    '0021500002',
    '0021500011',
    '0021500046',
    '0021500094',
    '0021500106',
    '0021500130',
    '0021500160',
    '0021500203',
    '0021500219',
    '0021500227'
]

test_game_ids = [
    '0021500313',
    '0021500367',
    '0021500405',
    '0021500424',
    '0021500438',
    '0021500453',
    '0021500473',
    '0021500543'
]

game_ids = [
    '0021500002',
    '0021500011',
    '0021500046',
    '0021500094',
    '0021500106',
    '0021500130',
    '0021500160',
    '0021500203',
    '0021500219',
    '0021500227',
    '0021500313',
    '0021500367',
    '0021500405',
    '0021500424',
    '0021500438',
    '0021500453',
    '0021500473',
    '0021500543'
]

df_list = []
possession_id_list_map = {}
for game_id in game_ids:
    df = pd.read_csv('./actions/{}_actions.csv'.format(game_id))
    possession_id_list = list(df['PossessionId'].unique())
    possession_id_list_map[game_id] = possession_id_list
    df_list.append(df)

df = pd.concat(df_list, axis = 0).reset_index()
df['SHOT_LOC_X'] = df['SHOT_LOC_X'].fillna(-500)
min_value = df['SHOT_LOC_X'].min()
max_value = df['SHOT_LOC_X'].max()
df['SHOT_LOC_X' + '_normalized'] = (df['SHOT_LOC_X'] - min_value) / (max_value - min_value)
df['SHOT_LOC_Y'] = df['SHOT_LOC_Y'].fillna(-100)
min_value = df['SHOT_LOC_Y'].min()
max_value = df['SHOT_LOC_Y'].max()
df['SHOT_LOC_Y' + '_normalized'] = (df['SHOT_LOC_Y'] - min_value) / (max_value - min_value)

df['2pt_shot_percentage_ball_handler'] = df['2pt_shot_percentage_ball_handler'].fillna(0.0)
df['3pt_shot_percentage_ball_handler'] = df['3pt_shot_percentage_ball_handler'].fillna(0.0)
df['2pt_shot_percentage_player1'] = df['2pt_shot_percentage_player1'].fillna(0.0)
df['3pt_shot_percentage_player1'] = df['3pt_shot_percentage_player1'].fillna(0.0)
df['2pt_shot_percentage_shot_player'] = df['2pt_shot_percentage_shot_player'].fillna(0.0)
df['3pt_shot_percentage_shot_player'] = df['3pt_shot_percentage_shot_player'].fillna(0.0)
df['game_clock_normalized'] = (720.0 - df['game_clock']) / 720.0

df['position_ball_handler'] = df['position_ball_handler'].fillna('no')
df['position_player1'] = df['position_player1'].fillna('no')
df['position_shot_player'] = df['position_shot_player'].fillna('no')
df['PossessionStartMarginScore_bin'] = np.select(
    [
        df['PossessionStartMarginScore'] < -6,
        (df['PossessionStartMarginScore'] >= -6) & (df['PossessionStartMarginScore'] < 0),
        df['PossessionStartMarginScore'] == 0,
        (df['PossessionStartMarginScore'] <= 6) & (df['PossessionStartMarginScore'] > 0),
        df['PossessionStartMarginScore'] > 6
    ],
    [
        'big_behind',
        'small_behing',
        'equal',
        'small_leading',
        'big_leading'
    ]
)

df = pd.get_dummies(df, columns = [
    'position_ball_handler', 'position_player1' , 'position_shot_player', 'PossessionStartMarginScore_bin'])

useful_feature_name = [
    '2pt_shot_percentage_ball_handler',
    '3pt_shot_percentage_ball_handler',
    '2pt_shot_percentage_player1',
    '3pt_shot_percentage_player1',
    '2pt_shot_percentage_shot_player',
    '3pt_shot_percentage_shot_player',
    'game_clock_normalized',
    'SHOT_LOC_X_normalized',
    'SHOT_LOC_Y_normalized', 
    'game_clock_normalized',
    'position_ball_handler_C', 
    'position_ball_handler_C-F',
    'position_ball_handler_F', 
    'position_ball_handler_G',
    'position_ball_handler_G-F', 
    'position_player1_C',
    'position_player1_C-F', 
    'position_player1_F', 
    'position_player1_G',
    'position_player1_G-F', 
    'position_player1_no', 
    'position_shot_player_C',
    'position_shot_player_C-F', 
    'position_shot_player_F',
    'position_shot_player_G', 
    'position_shot_player_G-F',
    'position_shot_player_no', 
    'PossessionStartMarginScore_bin_big_behind',
    'PossessionStartMarginScore_bin_big_leading',
    'PossessionStartMarginScore_bin_equal',
    'PossessionStartMarginScore_bin_small_behing',
    'PossessionStartMarginScore_bin_small_leading'
]

# for game_id in train_game_ids:
#     logger.info("Game Id: {}".format(game_id))
#     for possession_id in possession_id_list_map[game_id]:
#         logger.info("Possession Id: {}".format(possession_id))
#         episode_df = df[(df['PossessionId'] == possession_id) & (df['game_id'] == int(game_id))]
#         batch_flat_features = []
#         batch_image_features = []
#         batch_next_flat_features = []
#         batch_next_image_features = []
#         batch_actions = []
#         batch_rewards = []
#         for i in range(sequence_num-1, len(episode_df)):
#             logger.info('Global Step: {}'.format(global_step))
#             global_step += 1

#             flat_features = []
#             for feature in useful_feature_name:
#                 flat_features.append(torch.from_numpy(np.asarray([episode_df.iloc[i][feature]]).astype(np.float32)).unsqueeze(0))
#             flat_feature_input = torch.cat(flat_features, axis = 1)
#             batch_flat_features.append(flat_feature_input)

#             next_flat_features = []
#             if i == len(episode_df) - 1:
#                 batch_next_flat_features.append(None)
#             else:
#                 for feature in useful_feature_name:        
#                     next_flat_feature_tensor = torch.from_numpy(np.asarray([episode_df.iloc[i+1][feature]]).astype(np.float32)).unsqueeze(0)
#                     next_flat_features.append(next_flat_feature_tensor)
#                 next_flat_feature_input = torch.cat(next_flat_features, axis = 1)
#                 batch_next_flat_features.append(next_flat_feature_input)

#             images = []
#             for j in range(sequence_num-1, -1, -1):
#                 moment_id = episode_df.iloc[i-j]['moment_id']
#                 image_path = './img_copy/{}_processed/{}.png'.format(game_id, moment_id)
#                 image = Image.open(image_path).convert("RGB").resize((120, 60))
#                 image = np.asarray(image)
#                 images.append(torch.from_numpy(image.transpose((2, 0, 1)).astype(np.float32)).unsqueeze(0))
#             batch_image_features.append(torch.from_numpy(np.asarray(images)))
            
#             next_images = []
#             if i == len(episode_df) - 1:
#                 batch_next_image_features.append(None)
#             else:
#                 for j in range(sequence_num-1, -1, -1):
#                     next_moment_id = episode_df.iloc[i+1-j]['moment_id']
#                     next_image_path = './img_copy/{}_processed/{}.png'.format(game_id, next_moment_id)
#                     next_image = Image.open(next_image_path).convert("RGB").resize((120, 60))
#                     next_image = np.asarray(next_image)
#                     next_images.append(torch.from_numpy(next_image.transpose((2, 0, 1)).astype(np.float32)).unsqueeze(0))
#                 batch_next_image_features.append(torch.from_numpy(np.asarray(next_images)))
            
#             action = episode_df[episode_df['moment_id'] == moment_id]['action'].values[0]
#             action = action_dict[action]
#             reward = episode_df[episode_df['moment_id'] == moment_id]['Reward'].values[0]
#             action = torch.from_numpy(np.array([action])).unsqueeze(0)
#             reward = torch.from_numpy(np.array([reward])).unsqueeze(0)
#             batch_rewards.append(reward)
#             batch_actions.append(action)

#             if len(batch_flat_features) == batch_size or i == len(episode_df) - 1:

#                 if len(batch_flat_features) == 1:
#                     value_ = torch.zeros(len(batch_flat_features), device=device)
#                 else:
#                     non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch_next_flat_features)), device=device, dtype=torch.bool)
#                     batch_next_flat_feature_input = torch.cat([f for f in batch_next_flat_features if f is not None], axis=0).to(device)
#                     batch_next_image_feature_input = torch.cat([f for f in batch_next_image_features if f is not None], axis=1).to(device)
#                     value_ = torch.zeros(len(batch_flat_features), device=device)
#                     value_[non_final_mask] = target_q_net(batch_next_image_feature_input, batch_next_flat_feature_input, batch_next_image_feature_input.shape[1], sequence_num).max(1).values.detach()
                
                
#                 batch_image_feature_input = torch.cat(batch_image_features, axis=1).to(device)
#                 batch_flat_feature_input = torch.cat(batch_flat_features, axis=0).to(device)
#                 batch_action_input = torch.cat(batch_actions, axis=0).to(device)
#                 batch_reward_input = torch.cat(batch_rewards, axis=0).to(device)
                
#                 value = q_net(batch_image_feature_input, batch_flat_feature_input, batch_image_feature_input.shape[1], sequence_num).gather(1, batch_action_input)
#                 expected = gamma * value_.unsqueeze(1) + reward
#                 loss = F.smooth_l1_loss(value, expected)
#                 optimizer.zero_grad()
#                 loss.backward()
#                 for param in q_net.parameters():
#                     param.grad.data.clamp_(-1, 1)
#                 optimizer.step()
#                 running_loss += loss.item()
#                 batch_flat_features = []
#                 batch_image_features = []
#                 batch_next_flat_features = []
#                 batch_next_image_features = []
#                 batch_actions = []
#                 batch_rewards = []

#             if global_step % loss_plot_step == 0:
#                 logger.info("loss per 8 batches: {}".format(running_loss / 8))
#                 loss_values.append(running_loss / 8)
#                 show(loss_values, 8, "loss per 8 batches", "loss", "steps", "./loss_scale.png")
#                 running_loss = 0.0

#             if global_step % target_update_step == 0:
#                 target_q_net.load_state_dict(q_net.state_dict())
#                 torch.save(q_net.state_dict(), './dqn_basketball.pth')
    
# torch.save(q_net.state_dict(), './dqn_basketball.pth')



q_net.load_state_dict(torch.load('./dqn_basketball.pth'))

game_id = '0021500367' 
for possession_id in possession_id_list_map[game_id]:
    episode_df = df[(df['PossessionId'] == possession_id) & (df['game_id'] == int(game_id))]
    for i in range(sequence_num-1, len(episode_df)):

        flat_features = []
        for feature in useful_feature_name:
            flat_features.append(torch.from_numpy(np.asarray([episode_df.iloc[i][feature]]).astype(np.float32)).unsqueeze(0))
        flat_feature_input = torch.cat(flat_features, axis = 1)
        # print(flat_feature_input.shape)

        images = []
        for j in range(sequence_num-1, -1, -1):
            moment_id = episode_df.iloc[i-j]['moment_id']
            image_path = './img_copy/{}_processed/{}.png'.format(game_id, moment_id)
            image = Image.open(image_path).convert("RGB").resize((120, 60))
            image = np.asarray(image)
            images.append(torch.from_numpy(image.transpose((2, 0, 1)).astype(np.float32)).unsqueeze(0))
        image_feauture_input = torch.from_numpy(np.asarray(images))
        # print(image_feauture_input.shape)

        real_action = episode_df.iloc[i]['action']
        moment_id = episode_df.iloc[i]['moment_id']
        action = action_index_dict[q_net(image_feauture_input.to(device), flat_feature_input.to(device), 1, sequence_num).max(1).indices.item()]

        logger.info('Game Id: {}, Possession Id: {}, Moment Id: {}, Event Id: {}, Event Description: {}, Real Action: {}, Estimated Action: {}'.format(
            game_id,
            possession_id,
            moment_id,
            episode_df.iloc[i]['event_id'],
            episode_df.iloc[i]['EventDescription'],
            real_action,
            action
        ))


# image_path = './img_copy/0021500002_processed/{}.png'.format(1576)
# image = Image.open(image_path).convert("RGB").resize((120, 60)).save('test.png', quality=95)
# image = np.asarray(image)
# print(image.shape)

    
