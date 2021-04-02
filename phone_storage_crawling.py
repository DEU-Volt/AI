# -*- coding: utf-8 -*-

# 제목 : 번개장터 중고상품 휴대폰 범주 한정 (용량) 크롤링 소스
# 작성자 : 전규빈

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.support.ui import WebDriverWait
import os
from os import path

with webdriver.Edge('msedgedriver.exe') as driver:
    wait = WebDriverWait(driver, 1)

    folder_name = "train_image"
    if not path.isdir(folder_name):
        os.mkdir(folder_name)

    file_list = os.listdir(folder_name)
    file_number = len(file_list) + 1
    element_number = 1
    page_number = 1
    storage_types = [str(2 ** i) for i in range(4, 10)]
    ending_types = ['기', 'g', 'G', ' ', '블', ',']
    while True:
        try:
            driver.get(f"https://m.bunjang.co.kr/categories/600700001?page={page_number}&order=date")
        except Exception as e:
            print(e)
            break

        title_locator = (By.CSS_SELECTOR, f"div > a > div.sc-dznXNo.jFyxuP > div.sc-ekulBa.bPZorb")
        title_conditions = expected_conditions.presence_of_all_elements_located(title_locator)
        title_elements = wait.until(title_conditions, f"The title elements don't exist at this page {page_number}")

        for title_element in title_elements:
            title = title_element.text
            print(title)
            storage_found = []
            for storage_type in storage_types:
                storage_index = title.find(storage_type)
                if storage_index >= 0:
                    ending_index = storage_index + len(storage_type)
                    if len(title) <= ending_index:
                        storage_found.append(storage_type)
                    elif title[ending_index] in ending_types:
                        storage_found.append(storage_type)
            if len(storage_found) == 1:
                print(f"!!!!!!!!!!!!!!!!!! {storage_found} !!!!!!!!!!!!!!!!!!")
        input()

        page_number += 1

    input("press any key to exit")
