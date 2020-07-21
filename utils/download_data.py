"""File to download all the tfrecord and zip created at the moment with colab only.
Version: 1.0
Made by: Edgar Rangel
"""
import os
import subprocess

if os.getenv("SERVER") == '0':
    def parkinson_gait_cutted_zip(file_name:str = "data.zip"):
        """Function to download the dataset Colombian Parkinson Gait from drive in 
        the content folder. The file downloaded is a zip and will be unzipped after the
        download"""
        file_id = "1doRYrI2EjZlSpjcqYTB6YVfD_vkx76pR"
        subprocess.run(["bash", "/content/utils/download_from_drive.sh", 
            file_id, file_name])
        subprocess.run(["unzip", "-q", file_name, "-d", "/content/data/"])
        os.remove("/content/"+file_name)
        print("The dataset was downloaded, extracted and is ready to use."+
            "\nPath of the dataset: /content/data/")

elif os.getenv("SERVER") == '1':
    raise NotImplementedError("This module is not intended to work in the local "
        "server because the data is already in it. Just find it ;)")

else:
    raise ImportError("You can't import the download_data module because you "
        "didn't set up the env variable 'SERVER'")