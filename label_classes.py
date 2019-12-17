import csv
import os
import subprocess
import glob
import time
import re

makedirs = True

keys = ['/t/dd00125',	'/t/dd00126',	'/t/dd00127',	'/t/dd00128',	'/t/dd00129',	'/m/01b9nn',	'/m/01jnbd']
classes = ["INT-small-room", "INT-large-room-or-hall", "INT-public-space", "EXT-urban-or-manmade", "EXT-rural-or-natural", "Reverberation", "Echo"]

search_dir = './audio/raw_val/'
output_dir = './audio/val/'
files = list(filter(os.path.isfile, glob.glob(search_dir + "*")))

if makedirs:
    for c in classes:
        if not os.path.exists(output_dir + c):
            print('no directory named: ' + output_dir + c + " ...creating it")
            os.makedirs(output_dir + c)

url_id = []
for f in files:
    filename, ext = os.path.splitext(f)
    length = len(filename)
    id = filename[length-11:length]
    url_id.append((f, id))

links = []
lengths = []

total_to_match = len(files)

with open('./splits/eval_segments.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    unmatched = 0
    for row in csv_reader:
        for fid in url_id:
            file, id = fid
            if id == row[0]:
                total_to_match -= 1
                print("matches left " + str(total_to_match))

                num_classes = len(row) - 3
                all_classes = []
                for r in row:
                    r = re.sub('"', '', r)
                    r = re.sub(' ', '', r)
                    if r in keys:
                        all_classes.append(r)
                        print('found ' + r)

                if len(all_classes) == 0:
                    print('no match found, breaking')
                    unmatched += 1
                    break

                class_idx = keys.index(all_classes[0])
                resolved_class = classes[class_idx]
                # this_class = classes[]

                this_filename, ext = os.path.splitext(file)
                file_dir, filename_no_path = os.path.split(this_filename)
                new_filename = output_dir + resolved_class + '/' + filename_no_path + '.wav'
                print('creating new file: ' + new_filename)
                subprocess.run(["ffmpeg", "-i", file, "-ss", row[1], "-to", row[2], new_filename])

print("number of unmatched: " + str(unmatched))
