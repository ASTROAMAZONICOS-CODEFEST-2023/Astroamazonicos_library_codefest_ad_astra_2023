import os

class VideoAnalizer():
    
    def detect_objects_in_video(self, video_path: str, output_path: str):
        with open(output_path, 'w') as f:
            for frame in self.convert_frames(video_path, 1):
                frame_text = self.read_ocr(frame)
                coordinates = self.parse_coordinates(frame_text)
                time = self.parse_time(frame_text)

                objects = self.give_objects(frame)
                for obj in objects:
                    obj_id = obj['id']
                    obj_type = obj['type']
                    f.write(f'{obj_id},{obj_type},{time},{coordinates}\n')
                

    def convert_frames(self, video: str, framerate: float):
        pass

    def read_ocr(self, frame) -> str:
        pass

    def parse_coordinates(self, text: str) -> list:
        pass

    def parse_time(self, text: str) -> str:
        pass

    def give_objects(self, frame) -> list:
        pass

    def save_img(self, frame, obj: dict, output_path: str):
        # Create output directory if it doesn't exist
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            os.makedirs(f'{output_path}/IMG')

        cv2.imwrite(f'{output_path}/IMG/{id}.jpg', frame)
