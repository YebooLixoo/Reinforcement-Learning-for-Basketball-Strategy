import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.spatial import ConvexHull
import numpy as np
import time
import os


def draw_court(axis):
    import matplotlib.image as mpimg
    img = mpimg.imread('./court.png') # read image. I got this image from gmf05's github.
    plt.imshow(img,extent=axis, zorder=0) # show the image.

def sec2min(seconds):
    min = seconds // 60
    sec = seconds % 60
    return f"{min}:{sec}"

data_path = None
data = None

def draw(m_id):
    directory = f'./img_copy/{data_path.split("/")[-1].split(".")[0]}'
    os.makedirs(directory, exist_ok=True)
    ball_xy = data[data['moment_id']==m_id][["x_loc","y_loc","quarter", "game_clock","shot_clock"]].iloc[0].values
    player_xy = data[data['moment_id']==m_id][["jersey","x_loc","y_loc","abbreviation"]].iloc[1:].values
    fig = plt.figure(figsize=(15,7.5))
    ax = plt.gca()
    # frame_text1 = ax.text(0.505, 0.88, '', transform=ax.transAxes, ha='center', va='center')
    # frame_text2 = ax.text(0.445, 0.12, '', transform=ax.transAxes, ha='center', va='center')
    # frame_text1.set_text(f"PERIOD:    Q{int(ball_xy[2])}-{sec2min(round(ball_xy[3]))}") # quarter and game clock
    # frame_text1.set_fontsize(20)
    # frame_text2.set_text("SHOTCLOCK:   "+str(round(ball_xy[4],1))) # shot clock
    # frame_text2.set_fontsize(20)
    # team_a_name = player_xy[0][3]
    # team_b_name = player_xy[5][3]
    # team_a_handle = mpatches.Patch(color=(0.7, 0, 0, 1), label=team_a_name)
    # team_b_handle = mpatches.Patch(color=(0, 0, 0.7, 1), label=team_b_name)
    # ax.legend(handles=[team_a_handle, team_b_handle], loc='lower left', bbox_to_anchor=(0, -0.1), ncol=2)
    draw_court([0,100,0,50])
    for i in range(len(player_xy)):
        if player_xy[i][3] == 'CLE':
            # col = (0, 0, 0.7, 1)
            col = 'b'
        else:
            # col = (0.7, 0, 0, 1)
            col = 'r'
        ax.add_patch(plt.Circle((player_xy[i][1], player_xy[i][2]), 3, facecolor=col, edgecolor='none'))
        ax.text(player_xy[i][1], player_xy[i][2], str(int(player_xy[i][0])), color='w', ha='center', va='center', fontsize=30)

    ax.add_patch(plt.Circle((ball_xy[0], ball_xy[1]), 1.5, color='g'))

    # team_a_points = np.array([player_xy[i][1:3] for i in range(len(player_xy)) if i < 5])
    # hull_a = ConvexHull(team_a_points)
    # plt.fill(team_a_points[hull_a.vertices, 0], team_a_points[hull_a.vertices, 1], 'r', alpha=0.3, edgecolor='none')
    # team_b_points = np.array([player_xy[i][1:3] for i in range(len(player_xy)) if i >= 5])
    # hull_b = ConvexHull(team_b_points)
    # plt.fill(team_b_points[hull_b.vertices, 0], team_b_points[hull_b.vertices, 1], 'b', alpha=0.3, edgecolor='none')
    plt.axis('off')
    plt.savefig(f'{directory}/{m_id}.png', bbox_inches='tight', dpi=30)
    plt.close()

import multiprocessing

time1 = time.time()

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

for game_id in game_ids:
    data_path = "./processed_movement/{}_processed.csv".format(game_id)
    data = pd.read_csv(data_path)
    m_ids = data['moment_id'].unique()

    pool = multiprocessing.Pool()
    pool.map(draw, m_ids)
    pool.close()
    pool.join()

time2 = time.time()
print(time2-time1)

