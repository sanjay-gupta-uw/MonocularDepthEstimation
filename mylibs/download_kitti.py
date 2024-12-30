import os
import shutil
import zipfile
from typing import Any, Callable, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision.datasets.utils import download_url

data_url = "http://s3.eu-central-1.amazonaws.com/avg-kitti/"

city_resources = [ # 28 scenes from KITTI dataset
    "2011_09_26_drive_0001",
    "2011_09_26_drive_0002",
    "2011_09_26_drive_0005",
    "2011_09_26_drive_0009",
    "2011_09_26_drive_0011",
    "2011_09_26_drive_0013",
    "2011_09_26_drive_0014",
    "2011_09_26_drive_0017",
    "2011_09_26_drive_0018",
    "2011_09_26_drive_0048",
    "2011_09_26_drive_0051",
    "2011_09_26_drive_0056",
    "2011_09_26_drive_0057",
    "2011_09_26_drive_0059",
    "2011_09_26_drive_0060",
    "2011_09_26_drive_0084",
    "2011_09_26_drive_0091",
    "2011_09_26_drive_0093",
    "2011_09_26_drive_0095",
    "2011_09_26_drive_0096",
    "2011_09_26_drive_0104",
    "2011_09_26_drive_0106",
    "2011_09_26_drive_0113",
    "2011_09_26_drive_0117",
    "2011_09_28_drive_0001",
    "2011_09_28_drive_0002",
    "2011_09_29_drive_0026",
    "2011_09_29_drive_0071",
]
residential_resources = [
    "2011_09_26_drive_0019", # 21 scenes from KITTI dataset
    "2011_09_26_drive_0020",
    "2011_09_26_drive_0022",
    "2011_09_26_drive_0023",
    "2011_09_26_drive_0035",
    "2011_09_26_drive_0036",
    "2011_09_26_drive_0039",
    "2011_09_26_drive_0046",
    "2011_09_26_drive_0061",
    "2011_09_26_drive_0064",
    "2011_09_26_drive_0079",
    "2011_09_26_drive_0086",
    "2011_09_26_drive_0087",
    "2011_09_30_drive_0018",
    "2011_09_30_drive_0020",
    "2011_09_30_drive_0027",
    "2011_09_30_drive_0028",
    "2011_09_30_drive_0033",
    "2011_09_30_drive_0034",
    "2011_10_03_drive_0027",
    "2011_10_03_drive_0034",
]
road_resources = [
    "2011_09_26_drive_0015", # 12 scenes from KITTI dataset
    "2011_09_26_drive_0027",
    "2011_09_26_drive_0028",
    "2011_09_26_drive_0029",
    "2011_09_26_drive_0032",
    "2011_09_26_drive_0052",
    "2011_09_26_drive_0070",
    "2011_09_26_drive_0101",
    "2011_09_29_drive_0004",
    "2011_09_30_drive_0016",
    "2011_10_03_drive_0042",
    "2011_10_03_drive_0047",
]
campus_resources = [
    "2011_09_28_drive_0016", # 10 scenes from KITTI dataset
    "2011_09_28_drive_0021", 
    "2011_09_28_drive_0034", 
    "2011_09_28_drive_0035", 
    "2011_09_28_drive_0037", 
    "2011_09_28_drive_0038", 
    "2011_09_28_drive_0039", 
    "2011_09_28_drive_0043", 
    "2011_09_28_drive_0045", 
    "2011_09_28_drive_0047", 
]
person_resources = [
    "2011_09_28_drive_0053",  # 79 scenes from KITTI dataset
    "2011_09_28_drive_0054", 
    "2011_09_28_drive_0057", 
    "2011_09_28_drive_0065", 
    "2011_09_28_drive_0066", 
    "2011_09_28_drive_0068", 
    "2011_09_28_drive_0070", 
    "2011_09_28_drive_0071", 
    "2011_09_28_drive_0075", 
    "2011_09_28_drive_0077", 
    "2011_09_28_drive_0078",
    "2011_09_28_drive_0080",
    "2011_09_28_drive_0082",
    "2011_09_28_drive_0087",
    "2011_09_28_drive_0089",
    "2011_09_28_drive_0090",
    "2011_09_28_drive_0094",
    "2011_09_28_drive_0095",
    "2011_09_28_drive_0096",
    "2011_09_28_drive_0098",
    "2011_09_28_drive_0100",
    "2011_09_28_drive_0102",
    "2011_09_28_drive_0103",
    "2011_09_28_drive_0104",
    "2011_09_28_drive_0106",
    "2011_09_28_drive_0108",
    "2011_09_28_drive_0110",
    "2011_09_28_drive_0113",
    "2011_09_28_drive_0117",
    "2011_09_28_drive_0119",
    "2011_09_28_drive_0121",
    "2011_09_28_drive_0122",
    "2011_09_28_drive_0125",
    "2011_09_28_drive_0126",
    "2011_09_28_drive_0128",
    "2011_09_28_drive_0132",
    "2011_09_28_drive_0134",
    "2011_09_28_drive_0135",
    "2011_09_28_drive_0136",
    "2011_09_28_drive_0138",
    "2011_09_28_drive_0141",
    "2011_09_28_drive_0143",
    "2011_09_28_drive_0145",
    "2011_09_28_drive_0146",
    "2011_09_28_drive_0149",
    "2011_09_28_drive_0153",
    "2011_09_28_drive_0154",
    "2011_09_28_drive_0155",
    "2011_09_28_drive_0156",
    "2011_09_28_drive_0160",
    "2011_09_28_drive_0161",
    "2011_09_28_drive_0162",
    "2011_09_28_drive_0165",
    "2011_09_28_drive_0166",
    "2011_09_28_drive_0167",
    "2011_09_28_drive_0168",
    "2011_09_28_drive_0171",
    "2011_09_28_drive_0174",
    "2011_09_28_drive_0177",
    "2011_09_28_drive_0179",
    "2011_09_28_drive_0183",
    "2011_09_28_drive_0184",
    "2011_09_28_drive_0185",
    "2011_09_28_drive_0186",
    "2011_09_28_drive_0187",
    "2011_09_28_drive_0191",
    "2011_09_28_drive_0192",
    "2011_09_28_drive_0195",
    "2011_09_28_drive_0198",
    "2011_09_28_drive_0199",
    "2011_09_28_drive_0201",
    "2011_09_28_drive_0204",
    "2011_09_28_drive_0205",
    "2011_09_28_drive_0208",
    "2011_09_28_drive_0209",
    "2011_09_28_drive_0214",
    "2011_09_28_drive_0216",
    "2011_09_28_drive_0220",
    "2011_09_28_drive_0222",
]

# apply masking to the resources: 2011_09_26_drive_0015 -> raw_data/2011_09_26_drive_0015/2011_09_26_drive_0015_sync.zip
city_resources = [f"raw_data/{resource}/{resource}_sync.zip" for resource in city_resources]
residential_resources = [f"raw_data/{resource}/{resource}_sync.zip" for resource in residential_resources]
road_resources = [f"raw_data/{resource}/{resource}_sync.zip" for resource in road_resources]
campus_resources = [f"raw_data/{resource}/{resource}_sync.zip" for resource in campus_resources]
person_resources = [f"raw_data/{resource}/{resource}_sync.zip" for resource in person_resources]

image_dir_name_left = "image_02"  # Left images
image_dir_name_right = "image_03"  # Right images

raw_folder = os.path.join("../data/MyKitti/", "raw")

def check_exists(filename):
    if not os.path.exists(f"../data/MyKitti/raw/{image_dir_name_left}/{filename}"):
        return False
    return True


def DownloadKitti(root: str):
    download(city_resources)
    download(residential_resources)
    download(road_resources)
    download(campus_resources)
    download(person_resources)


def download(resources) -> None:
        """
        Download the KITTI data and extract only image_02 and image_03 folders.
        """
        os.makedirs(raw_folder, exist_ok=True)

        for fname in resources:
            filename = os.path.basename(fname)[:-4]
            # print(f"File name: {filename}")
            
            if check_exists(filename):
                # print("Files already downloaded and verified.")
                continue
            
            zip_path = os.path.join(raw_folder, os.path.basename(fname))
            print(f"Downloading {fname}...")
            download_url(
                url=f"{data_url}{fname}",
                root=raw_folder,
                filename=os.path.basename(fname),
            )
            print("Downloaded ZIP file.")

            # Use a temporary extraction directory
            tmp_dir = os.path.join(raw_folder, "temp")
            os.makedirs(tmp_dir, exist_ok=True)

            # Extract files to the temporary directory
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(tmp_dir)

            # Check for the presence of image_02 and image_03 files
            has_image_02 = any("image_02" in file and file.endswith(".png") for file in zip_ref.namelist())
            has_image_03 = any("image_03" in file and file.endswith(".png") for file in zip_ref.namelist())

            # Create target directories only if files exist
            if has_image_02:
                left_target = os.path.join(raw_folder, image_dir_name_left, filename)
                os.makedirs(left_target, exist_ok=True)
            if has_image_03:
                right_target = os.path.join(raw_folder, image_dir_name_right, filename)
                os.makedirs(right_target, exist_ok=True)

            # Move image_02 and image_03 files to their respective targets
            for root, _, files in os.walk(tmp_dir):
                for file in files:
                    if "image_02" in root and file.endswith(".png") and has_image_02:
                        shutil.move(os.path.join(root, file), os.path.join(left_target, file))
                    elif "image_03" in root and file.endswith(".png") and has_image_03:
                        shutil.move(os.path.join(root, file), os.path.join(right_target, file))

            # Clean up: remove the ZIP file and temporary directory
            os.remove(zip_path)
            shutil.rmtree(tmp_dir)
            print(f"Cleaned up {zip_path} and temporary files")


