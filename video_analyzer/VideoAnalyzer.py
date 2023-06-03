import numpy as np
import cv2
import easyocr
import imutils


class VideoAnalyzer():
    def __init__(self):
        """
        This function uses of entity labels from spacy to find dates. It also use the re library to find patterns in the text
        that could lead in to a date.
        input: 
        output: 
        """
        self.reader = easyocr.Reader(
            ["es", "en"], gpu=True)  # instance  Reader class, used for character recognition
        print("Reader easyocr class started")
        # initialize variables
        self.date = "NA"
        self.hour = "NA"
        self.coord1 = "NA"
        self.coord2 = "NA"
        self.id = 0
        self.object = "NA"

    def get_id(self):
        self.id += 1
        return self.id

    def detect_objects_in_video(self, video_path: str, output_path: str):
        with open(output_path, 'w') as f:  # start writing output
            f.write('id,type,time,coordinates\n')
            videocap = cv2.VideoCapture(video_path)
            framespersecond = int(videocap.get(cv2.CAP_PROP_FPS))
            for i in range(framespersecond):
                if i % 10 != 0: # skip frames because it is too slow
                    continue
                _, frame = videocap.read()  # get frame
                # call method that reads text from the frame and updates time, coordinates and date
                self.read_ocr(frame)
                if self.coord1 == "NA" or self.coord2 == "NA": # if coordinates are not found, skip frame
                    continue
                # call method that gets objects from the frame
                objects = self.give_objects(frame)
                for obj in objects:
                    obj_id = obj['id']
                    obj_type = obj['type']
                    detection = f'{obj_id},{obj_type},{self.hour},{self.coord1 + " - " + self.coord2}\n'
                    f.write(detection)

    def read_ocr(self, frame):
        """
        This function uses the easyocr library to read text from the frame and updates time, coordinates and date
        input: frame
        """
        result = self.reader.readtext(
            frame, paragraph=True)  # read text from image
        for res in result:
            text = res[1]  # get text
            chars = text.split(" ")  # Separate chars by spaces
            self.parse_time_date(chars)  # get time and date of the frame
            self.parse_coordinates(chars)  # get coordinates of the plane

    def parse_coordinates(self, chars: list):
        """
        This function uses the easyocr library to read text from the frame and updates time, coordinates and date
        input: chars
        """
        try:
            for i in range(len(chars)):
                if (len(chars[i]) > 10) and (len(chars[i+1]) > 10):  # Clasify chars by lenght
                    indx = len(chars[i])
                    self.coord1 = str(chars[i][indx-11:indx-10])+"°"+str(chars[i][indx-9:indx-7])+"'"+str(
                        chars[i][indx-6:indx-4])+"."+str(chars[i][indx-3:indx-1]) + '" N'  # Get first coordenate
                    self.coord2 = str(chars[i+1][indx-11:indx-9])+"°"+str(chars[i+1][indx-8:indx-6])+"'"+str(
                        chars[i+1][indx-5:indx-3])+"."+str(chars[i+1][indx-2:indx]) + '" W'  # Get second coordenate
        except:
            self.coord1 = "NA"
            self.coord2 = "NA"

    def parse_time_date(self, chars: list):
        """
        This function uses the easyocr library to read text from the frame and updates time, coordinates and date
        input: chars
        """
        for i in range(len(chars)):
            if (len(chars[i]) == 8):  # Clasify chars by lenght
                if ("/" in chars[i]):
                    self.date = str(
                        chars[i][0:2])+"/"+str(chars[i][3:5])+"/"+str(chars[i][6:8])  # Get date
                elif ("8" in chars[i]):
                    self.hour = str(
                        chars[i][0:2])+":"+str(chars[i][3:5])+":"+str(chars[i][6:8])  # Get time

    def give_objects(self, frame) -> list:
        """
        This function uses the contours of the image to find objects in the frame
        input: frame
        output: list of objects
        """

        img = np.asanyarray(frame)[:, :, ::-1].copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # apply thresholding to convert the grayscale image to a binary image
        _, thresh = cv2.threshold(gray, 50, 255, 0)

        # find the contours
        cnts, _ = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]

        # sort the contours by area and select maximum 10 contours
        cntsSorted = sorted(cnts, key=lambda x: cv2.contourArea(x))

        for _ in range(min(2, len(cntsSorted))):
            yield {
                'type': "VEHICULO",
                'id': self.get_id()
            }
