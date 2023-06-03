import os
import cv2
import numpy as np
import time #import time
from PIL import Image
import os
from os import system
import cv2
import easyocr
import time

class VideoAnalizer():
    def __init__(self):
        self.reader = easyocr.Reader(["es","en"], gpu=True) #instance  Reader class, used for character recognition
        print ("Reader easyocr class started")
        # initialize variables
        self.date = "NA"
        self.hour = "NA"
        self.coord1 = "NA"
        self.coord2 = "NA"
        self.id = "NA"
        self.object = "NA"

    def detect_objects_in_video(self, video_path: str, output_path: str):
        with open(output_path, 'w') as f: #start writing output
            for frame in self.convert_frames(video_path, 1):
                self.read_ocr(frame) # call method that reads text from the frame and updates time, coordinates and date
                objects = self.give_objects(frame) # call method that gets objects from the frame 
                for obj in objects:
                    obj_id = obj['id']
                    obj_type = obj['type']
                    f.write(f'{obj_id},{obj_type},{self.hour},{self.coord1 + " - " + self.coord2}, {self.date}\n')

    def convert_frames(self, video: str, framerate: int):
        comand = "ffmpeg -i "+ video + " -vf fps="+ str(framerate) + "Video/out%d.png" 
        system(comand)
    def read_ocr(self, frame):
        result = self.reader.readtext(frame, paragraph=True) #read text from image
        for res in result:
            text = res[1] #get text 
            chars = text.split(" ") #Separate chars by spaces
            self.parse_time_date(chars) # get time and date of the frame
            self.parse_coordinates(chars) # get coordinates of the plane    
    def parse_coordinates(self, chars: list):
        for i in  range( len(chars)):
            if (len(chars[i])>10) and (len(chars[i+1])>10): #Clasify chars by lenght
                indx = len(chars[i])
                self.coord1 = str(chars[i][indx-11:indx-10])+"°"+str(chars[i][indx-9:indx-7])+"'"+str(chars[i][indx-6:indx-4])+"."+str(chars[i][indx-3:indx-1]) +'" N' #Get first coordenate
                self.coord2 = str(chars[i+1][indx-11:indx-9])+"°"+str(chars[i+1][indx-8:indx-6])+"'"+str(chars[i+1][indx-5:indx-3])+"."+str(chars[i+1][indx-2:indx]) +'" W' #Get second coordenate

    def parse_time_date(self, chars: list):
        for i in  range( len(chars)):
            if (len(chars[i])==8): #Clasify chars by lenght
                if ("/" in chars[i]): 
                    self.date = str(chars[i][0:2])+"/"+str(chars[i][3:5])+"/"+str(chars[i][6:8]) #Get date
                elif ("8" in chars[i]):
                    self.hour = str(chars[i][0:2])+":"+str(chars[i][3:5])+":"+str(chars[i][6:8]) #Get time
                

    def give_objects(self, frame) -> list:
        pass

    def save_img(self, frame, obj: dict, output_path: str):
        # Create output directory if it doesn't exist
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            os.makedirs(f'{output_path}/IMG')

        cv2.imwrite(f'{output_path}/IMG/{id}.jpg', frame)
