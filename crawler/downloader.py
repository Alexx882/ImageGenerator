import requests
import os
import shutil

if __name__ == '__main__':
    if os.path.exists('images'):
        shutil.rmtree('images')
    if not os.path.exists('images'):
        os.mkdir('images')

    with open('crawler/sources.txt', 'r') as file:
        sources = file.readlines()
    
    for i in range(len(sources)):
        response = requests.get(sources[i])

        with open(f'images/{i}.png', 'wb') as file:
            file.write(response.content)
