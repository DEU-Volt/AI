# -*- coding: utf-8 -*-

# 제목 : 중고나라 중고상품 휴대폰 범주 한정 (제목, 가격) 크롤링 소스
# 작성자 : 전규빈

from selenium import webdriver
from selenium.webdriver import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.support.ui import WebDriverWait
from os import path
import time
import pandas as pd


with webdriver.Edge('msedgedriver.exe') as driver:
    driver.get(f"https://m.joongna.com/search-category?category=6")
    wait = WebDriverWait(driver, 1)
    element_number = 1

    file_name = 'train_data.csv'
    if not path.exists(file_name):
        df = pd.DataFrame({'title': [], 'price': []})
        df.to_csv(file_name, index=False, mode='w', encoding='utf-8-sig')

    while True:
        try:
            try:
                title_locator = (By.CSS_SELECTOR,
                                 f"#root > div.page > div:nth-child(2) > div:nth-child({element_number}) > div > div "
                                 f"> a:nth-child(1) > span")
                title_conditions = expected_conditions.presence_of_element_located(title_locator)
                title_elements = wait.until(title_conditions, f"The title elements don't exist.")


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
        try:
            price = int(price_elements.text.replace(',', '').replace('원', ''))
            df = pd.DataFrame({'title': [title], 'price': [price]})
            df.to_csv(file_name, header=False, index=False, mode='a', encoding='utf-8-sig')

        except Exception as e:
            print(e)
            continue
        print(element_number, title, price)

    input("press any key to exit")
