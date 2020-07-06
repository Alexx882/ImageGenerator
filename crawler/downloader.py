import requests
import os
import shutil

# number of images the downloader should download
N_IMAGES = 999999999

# number of images that should be skipped before counting
OFFSET = 0

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()


def run():
    if os.path.exists('images'):
        shutil.rmtree('images')
    if not os.path.exists('images'):
        os.mkdir('images')

    with open('crawler/sources.txt', 'r') as file:
        sources = file.readlines()

    n_total = min(len(sources), N_IMAGES+OFFSET)
    for i in range(n_total):
        if i < OFFSET:
            continue

        printProgressBar(iteration=i-OFFSET, total=n_total-1, prefix="Downloading: ")
        response = requests.get(sources[i])

        with open(f'images/{i}.png', 'wb') as file:
            file.write(response.content)


if __name__ == '__main__':
    run()