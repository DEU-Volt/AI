# -*- coding: utf-8 -*-

# 제목 : CNN 데이터 셋 만들기
# 설명 : 번개장터(이미지, 제목) 크롤링 소스
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
from konlpy.tag import Okt


def tokenizer_create(text):
    okt = Okt()
    text_pos = okt.pos(text, norm=True)

    words = []
    for word in text_pos:
        words.append(word[0])

    return words


with webdriver.Edge('msedgedriver.exe') as driver:
    wait = WebDriverWait(driver, 1)

    phone_list = {
        'lg': ['앨지', '엘지', 'LG'],
        'apple': ['아이폰', '애플', 'APPLE', 'IPHONE', '에플'],
        'samsung': ['갤럭시', '삼성', 'GALAXY', 'SAMSUNG', '겔럭시']
    }

    image_folder = "train_images"
    if not path.isdir(image_folder):
        os.mkdir(image_folder)
    for brand_folder in phone_list:
        if not path.isdir(f"{image_folder}/{brand_folder}"):
            os.mkdir(f"{image_folder}/{brand_folder}")

    page_number = 1
    samsung_count = 0
    apple_count = 0
    lg_count = 0
    while True:
        try:
            driver.get(f"https://m.bunjang.co.kr/categories/600700001?page={page_number}&order=date")
        except Exception as e:
            print(e)
            break

        title_locator = (By.CSS_SELECTOR, "div.sc-ciodno.LQqaE")
        image_locator = (By.CSS_SELECTOR, "div.sc-dznXNo.ezIRQX > img")

        title_conditions = expected_conditions.presence_of_all_elements_located(title_locator)
        image_conditions = expected_conditions.presence_of_all_elements_located(image_locator)

        title_elements = wait.until(title_conditions, f"The title elements don't exist at this page {page_number}")
        image_elements = wait.until(image_conditions, f"The image elements don't exist at this page {page_number}")

        for i in range(len(title_elements)):
            title = title_elements[i].text
            for word in tokenizer_create(title):
                word.upper()
                if lg_count < 1000 and word in phone_list['lg']:
                    brand_folder = "lg"
                    lg_count += 1
                    image_name = f"{image_folder}/{brand_folder}/{brand_folder}_{lg_count}.jpg"
                    break
                elif apple_count < 1000 and word in phone_list['apple']:
                    brand_folder = "apple"
                    apple_count += 1
                    image_name = f"{image_folder}/{brand_folder}/{brand_folder}_{apple_count}.jpg"
                    break
                elif samsung_count < 1000 and word in phone_list['samsung']:
                    brand_folder = "samsung"
                    samsung_count += 1
                    image_name = f"{image_folder}/{brand_folder}/{brand_folder}_{samsung_count}.jpg"
                    break
            else:
                continue

            image_src = image_elements[i].get_attribute('src')
            print(page_number, i, title, image_src)
            header = {'User-Agent': 'Edge/89.0.774.57'}
            result = requests.get(image_src, headers=header)
            image_bytes = io.BytesIO(result.content)
            image = Image.open(image_bytes)
            image = image.convert("RGB")
            image.save(image_name)

            if lg_count >= 1000 and samsung_count >= 1000 and apple_count >= 1000:
                break

        if lg_count >= 1000 and samsung_count >= 1000 and apple_count >= 1000:
            break
        page_number += 1

    input("press any key to exit")
