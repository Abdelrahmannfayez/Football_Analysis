from utils import read_video, save_video, get_center_of_bbox, get_bbox_width, get_foot_position
from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner  
import argparse
import cv2
from ultralytics import YOLO
import supervision as sv
import pickle
import os
import numpy as np
import pandas as pd
import sys 
import json
import os
import re
import cv2
import numpy as np


def get_team_in_possession(frame_num, team_ball_control):
    if frame_num < len(team_ball_control):
        team = team_ball_control[frame_num]
        if team == 1:
            return 'team-one'
        elif team == 2:
            return 'team-two'
    return 'no-team'

def update_json_with_team_info(json_path, output_json_path, team_ball_control):
    # Load JSON file
    with open(json_path, 'r') as file:
        events = json.load(file)

    # Update each event with the team in possession
    for event in events:
        frame_num = event['frame']
        team = get_team_in_possession(frame_num, team_ball_control)
        event['team'] = team

    # Save the updated JSON file
    with open(output_json_path, 'w') as file:
        json.dump(events, file, indent=4)

def extract_frame_number(filename):
    match = re.search(r'(\d+)', filename)
    if match:
        return int(match.group(1))
    return None

def process_image(image, frame_num, tracker, team_assigner, tracks, single_player_ball_control, first_frame):
    detection = tracker.model.predict(image, conf=0.1)
    cls_names = detection[0].names
    cls_names_inv = {v: k for k, v in cls_names.items()}
    detection_supervision = sv.Detections.from_ultralytics(detection[0])

    for object_ind, class_id in enumerate(detection_supervision.class_id):
        if cls_names[class_id] == "goalkeeper":
            detection_supervision.class_id[object_ind] = cls_names_inv["player"]

    detection_with_tracks = tracker.tracker.update_with_detections(detection_supervision)
    tracks["players"].append({})
    tracks["referees"].append({})
    tracks["ball"].append({})

    for frame_detection in detection_with_tracks:
        bbox = frame_detection[0].tolist()
        cls_id = frame_detection[3]
        track_id = frame_detection[4]

        if cls_id == cls_names_inv['player']:
            tracks["players"][frame_num][track_id] = {"bbox": bbox}

        if cls_id == cls_names_inv['referee']:
            tracks["referees"][frame_num][track_id] = {"bbox": bbox}

    for frame_detection in detection_supervision:
        bbox = frame_detection[0].tolist()
        cls_id = frame_detection[3]

        if cls_id == cls_names_inv['ball']:
            tracks["ball"][frame_num][1] = {"bbox": bbox}

    if frame_num == 0:
        first_frame = image
        team_assigner.assign_team_color(image, tracks['players'][0])

    player_track = tracks['players'][frame_num]
    for player_id, track in player_track.items():
        team = team_assigner.get_player_team(image, track['bbox'], player_id)
        tracks['players'][frame_num][player_id]['team'] = team
        tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]
        single_player_ball_control[player_id] = 0

    return first_frame

def main(input_folder, json_path):
    tracks = {
        "players": [],
        "referees": [],
        "ball": []
    }

    frame_num = 0
    team_assigner = TeamAssigner()
    first_frame = None
    single_player_ball_control = {}

    output_folder = 'Frames_OUT'
    output_json_path="jsonfile_OUT"
    if not os.path.exists(output_folder):
    os.makedirs(output_folder)

    tracker = Tracker('best.pt')

    frame_files = sorted([f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))], key=lambda x: extract_frame_number(x))

    for frame_file in frame_files:
        frame_path = os.path.join(input_folder, frame_file)
        frame = cv2.imread(frame_path)

        if frame is None:
            continue

        first_frame = process_image(frame, frame_num, tracker, team_assigner, tracks, single_player_ball_control, first_frame)
        frame_num += 1

    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])
    passes = {1: 0, 2: 0}
    ball_cut = {1: 0, 2: 0}
    previous_assigned_player = -1
    player_assigner = PlayerBallAssigner()
    team_ball_control = [0]

    for frame_num, player_track in enumerate(tracks['players']):
        ball_box = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_box)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            single_player_ball_control[assigned_player] += 1

            if assigned_player != previous_assigned_player and previous_assigned_player != -1:
                current_player_team = tracks['players'][frame_num][assigned_player]['team']
                previous_player_team = team_ball_control[-1]
                if current_player_team == previous_player_team:
                    passes[current_player_team] += 1
                else:
                    ball_cut[current_player_team] += 1

            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1])

        previous_assigned_player = assigned_player

    team_ball_control = np.array(team_ball_control)

    for frame_num, frame_file in enumerate(frame_files):
        frame_path = os.path.join(input_folder, frame_file)
        frame = cv2.imread(frame_path)

        if frame is None:
            continue

        frame = tracker.draw_frame_annotations(frame, frame_num, tracks, team_ball_control.copy())
        output_path = os.path.join(output_folder, frame_file)
        cv2.imwrite(output_path, frame)

        if (frame_num + 1) % 100 == 0:
            print(f'Processed {frame_num + 1} frames')
    
    update_json_with_team_info(json_path, output_json_path, team_ball_control)

    print('T1 passes:', passes[1])
    print('T2 passes:', passes[2])
    print('T1 cuts:', ball_cut[1])
    print('T2 cuts:', ball_cut[2])
    team1_possession = team_ball_control[team_ball_control == 1].shape[0]
    team2_possession = team_ball_control[team_ball_control == 2].shape[0]
    team1_percentage = (team1_possession / (team1_possession + team2_possession))
    team2_percentage = (team2_possession / (team1_possession + team2_possession))
    print('team1 possession', round(team1_percentage, 2))
    print('team2 possession', round(team2_percentage, 2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video frames and update JSON with team possession.")
    parser.add_argument('input_folder', type=str, help='Path to the folder containing input frames.')
    #parser.add_argument('output_folder', type=str, help='Path to the folder to save output frames.')
    parser.add_argument('json_path', type=str, help='Path to the input JSON file.')
    #parser.add_argument('output_json_path', type=str, help='Path to save the updated JSON file.')
    
    args = parser.parse_args()
    
    main(args.input_folder, args.json_path)
