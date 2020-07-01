import requests
import os
import shutil

# number of images the downloader should download
N_IMAGES = 999999999

# number of images that should be skipped before counting
OFFSET = 0

if __name__ == '__main__':
    if os.path.exists('images'):
        shutil.rmtree('images')
    if not os.path.exists('images'):
        os.mkdir('images')

    with open('crawler/sources.txt', 'r') as file:
        sources = file.readlines()
    
    for i in range(min(len(sources), N_IMAGES+OFFSET)):
        if i < OFFSET:
            continue

        print("downloading: "+sources[i])
        response = requests.get(sources[i])

        with open(f'images/{i}.png', 'wb') as file:
            file.write(response.content)
