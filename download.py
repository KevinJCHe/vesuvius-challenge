import os
import subprocess

from PIL import Image
from tqdm import tqdm
import numpy as np

import config # Contains ash2txt username and password

Image.MAX_IMAGE_PIXELS = None

# Download a set of files from the ash2txt server
def download_files(user, password, url, save_dir):
    print(url)
    # wget command
    cmd = [
        "wget",
        "--no-parent",
        "-r",
        "-l",
        "1",
        "--user=" + user,
        "--password=" + password,
        url,
        "-np",
        "-nd",
        "-nc",  # Don't overwrite files
        "-P",
        save_dir
    ]
    subprocess.run(cmd, check=True)

if __name__ == '__main__':
    # Download & prepare all fragment data
    fragment_ids = [
        '20231012184422_superseded',         
        '20231007101615_superseded',
        '20231016151001_superseded',
    ]
    # base_dir = r'C:\Users\Epicurus\Documents\Scrolls-Project\data\fragments'
    base_dir = '/mnt/c/Users/Epicurus/Documents/Scrolls-Project/data/fragments'
    base_url = 'http://dl.ash2txt.org/full-scrolls/Scroll1.volpkg/paths'
    for fragment_id in fragment_ids:
        output_path = f"{base_dir}/{fragment_id.replace('_superseded', '')}"
        os.makedirs(f'{output_path}/layers', exist_ok=True)
        
        # Download fragment metadata
        download_files(
            user=config.username,
            password=config.password,
            url=f'{base_url}/{fragment_id}/',
            save_dir=output_path,
        )

        # Download fragment layers
        download_files(
            user=config.username,
            password=config.password,
            url=f'{base_url}/{fragment_id}/layers/',
            save_dir=f'{output_path}/layers',
        )