# Astroamazonicos_library_codefest_ad_astra_2023
This repository contains the work done by the Astroamazonicos team during the CodeFest Ad Astra 2023 competition
# VideoAnalyzer library:
## Description:
The VideoAnalizer library contains methods inside `VideoAnalyzer` class that recives image and video data for analyzing information of Amazonas objects from an aerial perspective.

The main method is called `detec_objects_in_video`, and it creates a csv file containing an unique ID for each object detected, the type of a detected object, and the time registered in the video, the current coordinate of the plane that captures the video and the date associated to the video.
## Methods contained in the class:

### `__init__`
    method that constructs the class, in this method we create the arguments that are passed to the other methods and will be written in the csv file 
### `detect_objects_in_video(video_path: str, output_path: str)`

Method that receives the path of the video to be analyzed and the path where the csv file will be saved. This method reads the video frame per frame and calls the `read_ocr` method to extract time and coordinates. It also calls `give_objects` method to extract the objects detected in the frame.


### `read_ocr(frame)`

Method that receives an image, and using the easyOCR library, extracts all the characters identified in the input frame. The method separates the characters by spaces, and calls the `parse_time_date` method, and the `parse_coordinates` method. These will return the time and coordinates from the input frame respectively.

### `parse_coordinates(chars: list)`
Method that receives a list of characters. The method checks the length of each object in the list and since the coordinates allways have the same length and structure, parses key characters and deduces both coordinates wanted. The method updates the class atributes `coord1` and `coord2`.

### `parse_time_date(chars: list)`
Method that parses a time and date of each element in the  input list of characters. Since the time and date allways have the same length and structure, it is possible to parse them by indexing key characters. The method updates the class atributes "date" and "hour".

### `give_objects(frame)`

Method that receives an image and uses opencv to extract the maximum contours in the image. It contains a placeholder for object type, as the neural network is work in progress. The method returns a list of objects detected in the input frame.
