#/bin/bash

# Load train data
python batch_load_alfred_data.py \
--output_folder ./alfred_instance_data \
--alfred_dir /disk1/data/alfred/alfred_pc_V4/json_2.1.0 \
--alfred_names_file /disk1/data/alfred/alfred_pc_V4/json_2.1.0/meta_data/train.txt \
--need_rgb 1

# Load valid data
python batch_load_alfred_data.py \
--output_folder ./alfred_instance_data \
--alfred_dir /disk1/data/alfred/alfred_pc_V4/json_2.1.0 \
--alfred_names_file /disk1/data/alfred/alfred_pc_V4/json_2.1.0/meta_data/valid_seen.txt \
--need_rgb 1 \
--test 1

# Load valid data
python batch_load_alfred_data.py \
--output_folder ./alfred_instance_data \
--alfred_dir /disk1/data/alfred/alfred_pc_V4/json_2.1.0 \
--alfred_names_file /disk1/data/alfred/alfred_pc_V4/json_2.1.0/meta_data/valid_unseen.txt \
--need_rgb 1 \
--test 1

# Load test data
python batch_load_alfred_data.py \
--output_folder ./alfred_instance_data \
--alfred_dir /disk1/data/alfred/alfred_pc_V4/json_2.1.0 \
--alfred_names_file /disk1/data/alfred/alfred_pc_V4/json_2.1.0/meta_data/tests_seen.txt \
--need_rgb 1 \
--test 1

# Load test data
python batch_load_alfred_data.py \
--output_folder ./alfred_instance_data \
--alfred_dir /disk1/data/alfred/alfred_pc_V4/json_2.1.0 \
--alfred_names_file /disk1/data/alfred/alfred_pc_V4/json_2.1.0/meta_data/tests_unseen.txt \
--need_rgb 1 \
--test 1
