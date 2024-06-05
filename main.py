from utils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner  
import cv2
from ultralytics import YOLO
import supervision as sv
import pickle
import os
import numpy as np
import pandas as pd
import cv2
import sys 
sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width, get_foot_position


def main():


  tracks={
              "players":[],
              "referees":[],
              "ball":[]
          }

  frame_num = 0
  team_assigner = TeamAssigner()

  first_frame = None
  single_player_ball_control = {}


  video_path = '/content/drive/MyDrive/editing_project/coded_football/input_videos/football_vid.mp4'

  tracker = Tracker('/content/drive/MyDrive/editing_project/coded_football/models/best.pt')

  cap = cv2.VideoCapture(video_path)
  while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
      break
  
    detection = tracker.model.predict(frame,conf=0.1)

    # --------------------------------
    cls_names = detection[0].names
    cls_names_inv = {v:k for k,v in cls_names.items()}

    # Covert to supervision Detection format
    detection_supervision = sv.Detections.from_ultralytics(detection[0])
    # print(detection_supervision)
    # return 0
    # Convert GoalKeeper to player object
    for object_ind , class_id in enumerate(detection_supervision.class_id):
        if cls_names[class_id] == "goalkeeper":
            detection_supervision.class_id[object_ind] = cls_names_inv["player"]

    # Track Objects
    detection_with_tracks = tracker.tracker.update_with_detections(detection_supervision)

    tracks["players"].append({})
    tracks["referees"].append({})
    tracks["ball"].append({})

    for frame_detection in detection_with_tracks:
        bbox = frame_detection[0].tolist()
        cls_id = frame_detection[3]
        track_id = frame_detection[4]

        if cls_id == cls_names_inv['player']:
            tracks["players"][frame_num][track_id] = {"bbox":bbox}
        
        if cls_id == cls_names_inv['referee']:
            tracks["referees"][frame_num][track_id] = {"bbox":bbox}
    
    for frame_detection in detection_supervision:
        bbox = frame_detection[0].tolist()
        cls_id = frame_detection[3]

        if cls_id == cls_names_inv['ball']:
            tracks["ball"][frame_num][1] = {"bbox":bbox}

    # Assign Player Teams
    
    if frame_num == 0:
      first_frame = frame
      team_assigner.assign_team_color(frame, 
                                  tracks['players'][0])

    player_track = tracks['players'][frame_num]
    for player_id, track in player_track.items():
          team = team_assigner.get_player_team(frame,   
                                                track['bbox'],
                                                player_id)
          tracks['players'][frame_num][player_id]['team'] = team 
          tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]
          single_player_ball_control[player_id] = 0
    frame_num += 1
  cap.release()
 
          
  # Interpolate Ball Positions
  tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

  passes = {1:0,2:0}
  ball_cut = {1:0,2:0}
  previous_assigned_player = -1
  # assign ball possision
  player_assigner = PlayerBallAssigner()
  team_ball_control = [0]
  for  frame_num, player_track in enumerate(tracks['players']):
    ball_box = tracks['ball'][frame_num][1]['bbox']
    assigned_player = player_assigner.assign_ball_to_player(player_track,ball_box)
    
    

    if assigned_player != -1:
      tracks['players'][frame_num][assigned_player]['has_ball'] = True
      
      single_player_ball_control[assigned_player] += 1
     
      if assigned_player != previous_assigned_player and previous_assigned_player != -1 :
        current_player_team = tracks['players'][frame_num][assigned_player]['team']
        previous_player_team =team_ball_control[-1]
        if current_player_team == previous_player_team:
          passes[current_player_team] +=1
        else:
            ball_cut[current_player_team]+=1
         
      team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
    else:
      team_ball_control.append(team_ball_control[-1])

    previous_assigned_player = assigned_player

  team_ball_control= np.array(team_ball_control)
  
  
  frame_num = 0
  cap = cv2.VideoCapture(video_path)

    # Get video properties
  frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  fps = int(cap.get(cv2.CAP_PROP_FPS))
  output_video_path = '/content/drive/MyDrive/editing_project/coded_football/output_videos'
 
  # Define the codec and create VideoWriter object
  fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can use other codecs like 'XVID', 'MJPG', etc.
  out = cv2.VideoWriter("/content/drive/MyDrive/editing_project/coded_football/output_videos/output.avi", fourcc, fps, (frame_width, frame_height))

  while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
      break
   
    frame = tracker.draw_frame_annotations(frame,frame_num, tracks,team_ball_control.copy())

    out.write(frame)

    # Increment the frame count
    frame_num += 1

    # Optional: Print progress
    if (frame_num+1) % 100 == 0:
        print(f'Processed {frame_num+1} frames')

  # Release the video capture and writer objects
  cap.release()
  out.release()

# -------------------------------------------------------------------------------------------------------

  print('T1 passes:', passes[1])
  print('T2 passes:', passes[2])
  print('T1 cuts:', ball_cut[1])
  print('T2 cuts:', ball_cut[2])
  team1_possession = team_ball_control[team_ball_control==1].shape[0]
  team2_possession = team_ball_control[team_ball_control==2].shape[0]
  team1_percentage =  (team1_possession/(team1_possession+team2_possession))
  team2_percentage = (team2_possession/(team1_possession+team2_possession))
  print('team1 possession', round(team1_percentage,2))
  print('team2 possession', round(team2_percentage,2))

if __name__ == "__main__":
  main()
