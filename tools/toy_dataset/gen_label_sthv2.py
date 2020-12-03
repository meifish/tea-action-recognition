# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu
# ------------------------------------------------------
# Code adapted from https://github.com/metalbubble/TRN-pytorch/blob/master/process_dataset.py
# processing the raw data of the video Something-Something-V2

import os
import json
import os
from pathlib import Path

print(os.getcwd())
# Input
train_file = 'train.json'
validation_file = 'validation.json'
test_file = 'test.json'
label_file = 'labels.json'
category_file = 'category.txt'
frames = '/media/meiyu/0C9255199255091A/code/somethingsomething/toy_dataset/frames/'
# Output
val_videofolder = 'val_videofolder.txt'
train_videofolder = 'train_videofolder.txt'
test_videofolder = 'test_videofolder.txt'



if __name__ == '__main__':
    dataset_name = 'something-something-v2'  # 'jester-v1'
    with open(label_file) as f:
        data = json.load(f)
    categories = []
    for i, (cat, idx) in enumerate(data.items()):
        assert i+1 == int(idx)  # make sure the rank is right
        categories.append(cat)

    with open('category.txt', 'w') as f:
        f.write('\n'.join(categories))

    # import pdb; pdb.set_trace()
    dict_categories = {}
    for i, category in enumerate(categories):
        dict_categories[category] = i

    files_input = [validation_file, train_file]
    files_output = [val_videofolder, train_videofolder]
    for (filename_input, filename_output) in zip(files_input, files_output):
        with open(filename_input) as f:
            data = json.load(f)
        folders = []
        idx_categories = []
        for item in data:
            folders.append(item['id'])
            if 'test' not in filename_input:
                idx_categories.append(dict_categories[item['template'].replace('[', '').replace(']', '')])
            else:
                idx_categories.append(0)
        output = []
        for i in range(len(folders)):
            curFolder = folders[i]
            curIDX = idx_categories[i]
            # counting the number of frames in each video folders
            dir_files = os.listdir(os.path.join(frames, curFolder))
            output.append('%s %d %d' % (os.path.join(frames, curFolder), len(dir_files), curIDX))
            print('%d/%d' % (i, len(folders)))
        with open(filename_output, 'w') as f:
            f.write('\n'.join(output))
