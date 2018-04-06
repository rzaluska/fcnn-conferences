import glob
import os

from postprocess_fcnn_segmentation import postprocess_and_save

for file_path in sorted(glob.glob("frames/*.jpg")):
    print(file_path)
    filename = os.path.basename(file_path)
    filename = os.path.splitext(filename)[0]
    postprocess_and_save("predicted_frames/"+filename+".png", "frames/"+filename+".jpg", "final_output_frames/" + filename + ".jpg")
