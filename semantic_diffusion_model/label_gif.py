from PIL import Image
import os
import glob
import numpy as np

def make_gif(frame_folder, save_path, sample_num):
    frames = [Image.open(image) for image in sorted(glob.glob(f"{frame_folder}/*.png"))]    
    
    frame_one = frames[0]

    # drop every 10th frame until frame 900
    frames_until900 = frames[:900]
    frames_until900 = frames_until900[::10]
    frames = frames_until900 + frames[900:]
    frame_one.save(os.path.join(save_path, "example" + sample_num + ".gif"), format="GIF", append_images=frames,
               save_all=True, duration=1, loop=0)


# create a new folder for the images with the plus row
os.makedirs("/artifacts/tmp_plus", exist_ok=True)

# iterate through the images in the folder
directory = os.fsencode('./tmp')


#for file in os.listdir(directory):
#    filename = os.fsdecode(file)
#    if filename.endswith(".png"):
#        file_number = filename.split(".")[0]
'''
for file_number in range(1000):        

        # load file as image
        img = Image.open(f"./tmp/{file_number}.png")

        # add a plus row to the image
        img_plus = Image.new("RGB", (img.width, img.height + 10), (255, 255, 255))
        img_plus.paste(img, (0, 0))
        print(file_number)
        file_number = int(file_number)
        # color red for the plus row until 10% of the image width
        for i in range(int(img.width * (file_number + 1) / 1000)):
            for j in range(10):
                img_plus.putpixel((i, 256+j), (150, 0, 0))

        # put 0s to the left of the number
        if file_number < 10:
            file_number_str = "00" + str(file_number)
        elif file_number < 100:
            file_number_str = "0" + str(file_number)
        else:
            file_number_str = str(file_number)
        # save the image to ./tmp_plus/0.png
        img_plus.save(f"/artifacts/tmp_plus/{file_number_str}.png")
'''







make_gif("/artifacts/tmp_plus", "/artifacts/", str(0))