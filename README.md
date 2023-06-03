# Astroamazonicos_library_codefest_ad_astra_2023
This repository contains the work done by the Astroamazonicos team during the CodeFest Ad Astra 2023 competition
# VideoAnalizer library:
## Description:
The VideoAnalizer library contains methods that recive a video and gennerates a csv file containing an unic ID for each object detected, the type of object detected, the time registered in the video, the current coordinate of the plane that captures the video and the date associated to the video.
## Methods contained in the library:

### __init__
    method that constructs the class, in this method are created the arguments that are passed to the other methods and will be written in the txt file 
### detect_objects_in_video( video_path: str, output_path: str):


###  read_ocr( frame):

Method that receives an image, and usign the easyOCR library extracts all the characters identified in the input frame. The methot separates the characters by spaces and calls the parse_time_date method  and the parse_coordinates method, that are going to get the date, time and coordinates from the input frame.

###  parse_coordinates( chars: list):
Method that receives a list of characters. The method checks the length of each object in the list and since the coordinates allways have the same length and structure, index key characters and gets both coordinates wanted. The method updates the class atributes "coord1" and "coord2".


### parse_time_date( chars: list):
Method that parses a time and date of each element in the  input list of characters. Since the time and date allways have the same length and structure, it is possible to parse them by indexing key characters. The method updates the class atributes "date" and "hour".



