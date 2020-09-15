import requests
import os
import shutil
from typing import Iterable


def printProgressBar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
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

# TODO refactor to improve performance if needed
def retrieve_images(max_amount: int = 1000000, offset: int = 0, print_status: bool = False) -> Iterable:
    '''
    Retrieves images from the sources.txt
    
    :param max_amount:  maximal number of images to retrieve
    :param offset:      number of images to skip
    '''
    with open('crawler/sources.txt', 'r') as file:
        sources = file.readlines()

    n_total = min(len(sources), max_amount+offset)
    for i in range(n_total):
        if i < offset:
            continue

        if print_status:
            printProgressBar(iteration=i-offset, total=n_total-1, prefix="Downloading: ")
        response = requests.get(sources[i])

        yield response.content


def run():
    if os.path.exists('images'):
        shutil.rmtree('images')
    if not os.path.exists('images'):
        os.mkdir('images')

    cnt = 0
    for image in retrieve_images():
        with open(f'images/{cnt}.png', 'wb') as file:
            file.write(image)
        cnt += 1


if __name__ == '__main__':
    run()
    