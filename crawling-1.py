# -*- coding: utf-8 -*-

# 제목 : 번개장터 중고상품 (이미지, 제목, 가격 ) 크롤링 소스
# 작성자 : 전규빈

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.support.ui import WebDriverWait
import os
from os import path
import requests
from PIL import Image
import io

with webdriver.Edge('msedgedriver.exe') as driver:
    wait = WebDriverWait(driver, 1)

    folder_name = "train_image"
    if not path.isdir(folder_name):
        os.mkdir(folder_name)

    file_list = os.listdir(folder_name)
    file_number = len(file_list) + 1
    page_number = 1
    while True:
        try:
            driver.get(f"https://m.bunjang.co.kr/categories/600?page={page_number}")
        except Exception as e:
            print(e)
            break

        title_locator = (By.CSS_SELECTOR, "div.sc-ekulBa.bPZorb")
        image_locator = (By.CSS_SELECTOR, "div.sc-ghsgMZ.cFzedP > img")
        price_locator = (By.CSS_SELECTOR, "div.sc-ciodno")

        title_conditions = expected_conditions.presence_of_all_elements_located(title_locator)
        image_conditions = expected_conditions.presence_of_all_elements_located(image_locator)
        price_conditions = expected_conditions.presence_of_all_elements_located(price_locator)

        title_elements = wait.until(title_conditions, f"The title elements don't exist at this page {page_number}")
        image_elements = wait.until(image_conditions, f"The image elements don't exist at this page {page_number}")
        price_elements = wait.until(price_conditions, f"The price elements don't exist at this page {page_number}")

        for i in range(len(title_elements)):
            title = title_elements[i].text
            image_src = image_elements[i].get_attribute('src')
            try:
                price = int(price_elements[i].text.replace(',', ''))
            except Exception as e:
                print(e)
                continue
            print(page_number, i, title, image_src, price)

            header = {'User-Agent': 'Edge/89.0.774.57'}
            result = requests.get(image_src, headers=header)
            image_bytes = io.BytesIO(result.content)
            image = Image.open(image_bytes)
            image_size = (100, 100)
            image = image.resize(image_size)
            image = image.convert("RGB")
            image.save(f"{folder_name}/{file_number}_{price}.jpg")

            file_number += 1

        page_number += 1

    input("press any key to exit")
