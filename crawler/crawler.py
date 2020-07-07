import time
from typing import List, Iterator

import requests
from bs4 import BeautifulSoup
# download chromedriver and add to path
# https://sites.google.com/a/chromium.org/chromedriver/home
from selenium import webdriver
from selenium.webdriver.common.keys import Keys


URL = 'https://unsplash.com/s/photos/model'
NR_IMAGE_BATCHES = 500
SCROLL_AMOUNT = 5 # scroll amount for a single batch

def get_webpage(url, nr_batches) -> Iterator[str]:
    '''
    Loads the webpage in Chrome and scrolls down the endless scroller to load more images.
    :param url: URL to the page
    :param nr_batches: Amount of batches to return

    :returns: A iterator for html source with len = nr_batches
    '''
    driver = webdriver.Chrome()
    driver.get(url)
    time.sleep(1)

    body = driver.find_element_by_tag_name("body")

    for batch in range(nr_batches):
        yield driver.page_source

        for _ in range(SCROLL_AMOUNT):
            body.send_keys(Keys.PAGE_DOWN)
        time.sleep(2)

    driver.quit()


def write_output(image_urls: List[str]):
    with open('crawler/sources.txt', 'a') as file:
        file.write('\n'.join(image_urls))
        file.write('\n')


def run():
    bodies = get_webpage(URL, NR_IMAGE_BATCHES)

    for body in bodies:
        soup = BeautifulSoup(body, 'html.parser')
        img_tags = soup.findAll('img', class_='_2VWD4 _2zEKz')

        write_output([t['src'] for t in img_tags])

if __name__ == '__main__':
    run()
