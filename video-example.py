from video_analyzer.VideoAnalyzer import VideoAnalyzer
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('-i', '--input', help='path to video file')
argparser.add_argument('-o', '--output', help='path to output file')
args = argparser.parse_args()

# Create an instance of the VideoAnalyzer class
analyzer = VideoAnalyzer()

# Call the method that detects objects in the video
# Run this script with the following command:
# python video-example.py -i ../path/to/video -o ../path/to/output
analyzer.detect_objects_in_video(args.input, args.output)
