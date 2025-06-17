import cv2 
from PIL import Image
import os

class VideoTool(object):
    def __init__(self, video_pth):
        self.vid_pth = video_pth
        self.frame_pth = video_pth[:-4]
        pass

    def extract_frame(self):
        if(os.path.exists(self.frame_pth)):
            pass
        else:
            os.makedirs(self.frame_pth)

        cap = cv2.VideoCapture(self.vid_pth)
        frame_count = 0
        lst_frame = []
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            pil_image.save(os.path.join(self.frame_pth, f'frame_{frame_count:04d}.jpg'))
            lst_frame.append(pil_image)
            frame_count += 1
        cap.release()
        print(len(lst_frame))
        return 
    
if __name__ == "__main__":
    vid_pth = "/Users/duypd/MyPC/MyProject/SG-Retrieval/Datasets/Video/VID_20250405_104556.mp4"
    vid = VideoTool(vid_pth)
    vid.extract_frame()