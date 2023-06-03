# Astroamazonicos_library_codefest_ad_astra_2023
This repository contains the work done by the Astroamazonicos team during the CodeFest Ad Astra 2023 competition

# Prerequirements:
In order to use the library, required dependencies listed on the requirements.txt file.

Use the following commands:

```
python3 -m pip install -r requirements.txt
```

# Examples:

## `video-example.py`

This example shows how to use the VideoAnalyzer class to analyze a video and create a csv file with the information of the objects detected in the video.

It requires a video file to run.

The script shoulw be run as follows:
```
python3 video-example.py -i <path_to_video> -o <path_to_output_csv_file>
```

It will take a lot of time to process the video
> Aproxiamtely the same time as the video duration


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


# NLP library:
## Description:
This objective consists of identifying entities in the news that describe environmental effects on the Colombian Amazon. The available data set is made up of news from 2010 to 2023 and has 187 records, which describe situations affecting the Colombian Amazon related to contamination, illegal extraction of minerals and deforestation. In this type of news, it is important to automatically identify the actors that register some type of direct or indirect participation in said affectation. For this automated search and classification of texts, there are NER tools that identify entities such as locations, organizations, people and other actors in texts. Examples of these tools are: Spacy, BERT Google, NLP Stanford, Gate, OpenNLP, among others.

## Approach:
The approximation that was taken seeks to obtain information from four main sources:
1. A text directly
2. A text file
3. A data frame
4. A link to a news item, in which beautiful soup is used to be able to do webscraping and extract information from the text

Subsequently, the code undergoes a cleanup of the text, which involves removing stopwords, punctuation marks, putting all words in lowercase, and removing special characters. Then, spacy and regular expressions are used to extract entities such as: location, people, dates, important factors, a summary of the text and organizations.

## Usage:
Create a python module and import the library as follows
```
from NewsAnalyzer import NewsAnalyzer
```

Then an instance of the class is created as follows:
```
n = NewsAnalyzer()
```

Now choose the method of your choice to run, depending on the type of data you wish to put as input.

- ner_from_str(text, output_path)
- ner_from_file(text_path, output_path)
- ner_from_url(url, output_path)
- ner_from_df(path, output_path)

The result will be saved in output path.

** Disclaimer: ** To see all the features in depth see the working jupyter notebook file.
