# Import everything needed to edit video clips
from moviepy.editor import *
	
video_path = '/home/ducanh/hain/dataset/175.mp4'
save_path = '/home/ducanh/hain/dataset/175_30s.mp4'

# loading video dsa gfg intro video
clip = VideoFileClip(video_path)
	
# getting only first 5 seconds
clip = clip.subclip(0, 30)

# saving the clip
clip.write_videofile(save_path)