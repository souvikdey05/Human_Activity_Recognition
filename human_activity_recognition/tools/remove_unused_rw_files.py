#!/usr/bin/python3

''' Script to delete files from the realWorld dataset which are not used in the project.
This reduces the size of the dataset from ca. 4GB to ca. 2.4GB.

These files are:
    - The '/images' and '/videos' directories of each proband
    - Sensor data files not produced by accelerometer or gyroscope

Place this file in the 'realworld2016_dataset' directory, next to the 'proband<NUMBER>' directories,
and run it with 'python3 remove_unused_rw_files.py'.

Used Python version: 3.8
Used packages: pathlib and shutil.
'''

import pathlib
import shutil


proband_directories = [path for path in pathlib.Path.cwd().iterdir()
                            if path.is_dir() and path.stem.startswith('proband') ]

for proband_directory in proband_directories:

    image_directory = proband_directory / 'images'
    if image_directory.exists():
        print(f"Deleting directory: '{image_directory}'.")
        shutil.rmtree(path=image_directory, ignore_errors=True)

    video_directory = proband_directory / 'videos'
    if video_directory.exists():
        print(f"Deleting directory: '{video_directory}'.")
        shutil.rmtree(path=video_directory, ignore_errors=True)

    data_directory = proband_directory / 'data'
    unused_data_files = [path for path in data_directory.iterdir()
                            if not path.name.startswith( ('acc_', 'gyr_') ) ]

    for data_file in unused_data_files:
        print(f"Deleting file: '{data_file}'.")
        data_file.unlink(missing_ok=True)
