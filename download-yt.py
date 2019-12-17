import csv
import os
import subprocess

keys = ['/t/dd00125',	'/t/dd00126',	'/t/dd00127',	'/t/dd00128',	'/t/dd00129',	'/m/01b9nn',	'/m/01jnbd']
print(keys)
links = []

with open('./splits/balanced_train_segments.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        matching = [s for s in row if any(xs in s for xs in keys)]
        if len(matching) > 0:
            # print(row)
            print('www.youtube.com/watch?v=' + row[0])
            print(row[1])
            print(row[2])
            links.append('http://www.youtube.com/watch?v=' + row[0])
            line_count += 1
    print(f'Processed {line_count} lines.')

links = links[0:5]

for l in links:
    # command = "youtube-dl --extract-audio '" + l + "'"
    # print(command)
    # os.system("youtube-dl")

        # -o '%(title)s.%(ext)s
    result = subprocess.Popen(["youtube-dl", "--extract-audio", l],stdout=subprocess.PIPE)
    result.wait()
    print("NEW LINE NEW LINE NEW LINE NEW LINE NEW LINE NEW LINE NEW LINE NEW LINE ")
    print(result.communicate()[0])
    # out, err = result.communicate()/
    # print(out)
    # subprocess.run(["ffmpeg", "-i", ,"--ss", 10])

    # fmpeg -i input.mp3 -ss 10 -t 6 -acodec copy output.mp3
    # ffmpeg -ss 10 -t 6 -i input.mp3 output.mp3
