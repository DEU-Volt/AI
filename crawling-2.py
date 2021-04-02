# -*- coding: utf-8 -*-

# 제목 : 중고나라 중고상품 (이미지, 제목, 가격 ) 크롤링 소스
# 작성자 : 전규빈

from selenium import webdriver
from selenium.webdriver import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.support.ui import WebDriverWait
import os
from os import path
import time
import requests
import io
from PIL import Image

with webdriver.Edge('msedgedriver.exe') as driver:
    driver.get(f"https://m.joongna.com/search-category?category=7")

    folder_name = "train_image"
    if not path.isdir(folder_name):
        os.mkdir(folder_name)

    element_number = 1
    file_list = os.listdir(folder_name)
    file_number = len(file_list) + 1
    wait = WebDriverWait(driver, 1)
    while True:
        try:
            try:
                title_locator = (By.CSS_SELECTOR,
                                 f"#root > div.page > div:nth-child(2) > div:nth-child({element_number}) > div > div "
                                 f"> a:nth-child(1) > span")
                title_conditions = expected_conditions.presence_of_element_located(title_locator)
                title_elements = wait.until(title_conditions, f"The title elements don't exist.")

                image_locator = (By.CSS_SELECTOR,
                                 f"#root > div.page > div:nth-child(2) > div:nth-child({element_number}) > div > a > "
                                 f"div > img")
                image_conditions = expected_conditions.presence_of_element_located(image_locator)
                image_elements = wait.until(image_conditions, f"The image elements don't exist.")

                price_locator = (By.CSS_SELECTOR,
                                 f"#root > div.page > div:nth-child(2) > div:nth-child({element_number}) > div > div "
                                 f"> a:nth-child(1) > p")
                price_conditions = expected_conditions.presence_of_element_located(price_locator)
                price_elements = wait.until(price_conditions, f"The price elements don't exist.")
            except Exception as e:
                print(e)

                more_result_locator = (By.CSS_SELECTOR,
                                       "#root > div.page > div:nth-child(2) > "
                                       "div.pd_h20.pd_v15.SearchMoreButton_moreButtonWrap__3yMmV > button")
                more_result_conditions = expected_conditions.presence_of_element_located(more_result_locator)
                more_result_elements = wait.until(more_result_conditions, f"The button elements don't exist ")

                actions = ActionChains(driver)
                actions.move_to_element(more_result_elements)
                actions.click(more_result_elements)
                actions.perform()

                time.sleep(3)
                continue
        except Exception as e:
            print(e)
            break
        element_number += 1

        title = title_elements.text
        image_src = image_elements.get_attribute('src')
        try:
            price = int(price_elements.text.replace(',', '').replace('원', ''))
        except Exception as e:
            print(e)
            continue
        print(element_number, title, image_src, price)

        header = {'User-Agent': 'Edge/89.0.774.57'}
        result = requests.get(image_src, headers=header)
        image_bytes = io.BytesIO(result.content)
        image = Image.open(image_bytes)
        image_size = (100, 100)
        image = image.resize(image_size)
        image = image.convert("RGB")
        image.save(f"{folder_name}/{file_number}_{price}.jpg")
        file_number += 1
    input("press any key to exit")
