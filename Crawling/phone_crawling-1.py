# -*- coding: utf-8 -*-

# 제목 : 번개장터 중고상품 휴대폰 범주 한정 (제목, 가격) 크롤링 소스
# 작성자 : 전규빈


from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.support.ui import WebDriverWait
from os import path
import pandas as pd

with webdriver.Edge('msedgedriver.exe') as driver:
    wait = WebDriverWait(driver, 1)

    file_name = 'train_data.csv'
    if not path.exists(file_name):
        df = pd.DataFrame({'title': [], 'price': []})
        df.to_csv(file_name, index=False, mode='w', encoding='utf-8-sig')

    page_number = 1
    while True:
        try:
            driver.get(f"https://m.bunjang.co.kr/categories/600700001?page={page_number}&order=date")
        except Exception as e:
            print(e)
            break

        title_locator = (By.CSS_SELECTOR, "div.sc-ciodno.LQqaE")
        price_locator = (By.CSS_SELECTOR, "div.sc-gGCbJM.dZFstm")

        title_conditions = expected_conditions.presence_of_all_elements_located(title_locator)
        price_conditions = expected_conditions.presence_of_all_elements_located(price_locator)

        title_elements = wait.until(title_conditions, f"The title elements don't exist at this page {page_number}")
        price_elements = wait.until(price_conditions, f"The price elements don't exist at this page {page_number}")

        for i in range(len(title_elements)):
            title = title_elements[i].text
            try:
                price = int(price_elements[i].text.replace(',', ''))
                df = pd.DataFrame({'title': [title], 'price': [price]})
                df.to_csv(file_name, header=False, index=False, mode='a', encoding='utf-8-sig')
            except Exception as e:
                print(e)
                continue
            print(page_number, i, title, price)

        page_number += 1

    input("press any key to exit")
