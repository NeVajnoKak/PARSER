import time
import csv
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

# Настроим вебдрайвер (в данном случае для Chrome)
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

# Открываем страницу Bloomberg с данными
url = "https://www.bloomberg.com/markets/stocks/world-indexes/americas"
driver.get(url)

# Ждем, пока страница полностью загрузится (можно также использовать WebDriverWait)
time.sleep(5)  # Подождем 5 секунд, чтобы все динамически загружаемые элементы успели появиться

# Теперь находим таблицу
rows = driver.find_elements(By.XPATH, "//tr[@class='data-table-row']")

# Заголовки таблицы
headers = [
    "Name",
    "Value",
    "Net Change",
    "% Change",
    "1 Month",
    "1 Year",
    "Time (EST)"
]

# Список для хранения данных
data = []

# Извлекаем данные из каждой строки таблицы
for row in rows:
    # Извлекаем все ячейки в строке
    columns = row.find_elements(By.XPATH, ".//td | .//th[@class='data-table-row-cell']")

    if len(columns) >= 7:  # Если в строке есть все необходимые столбцы
        name = columns[0].text.strip()
        value = columns[1].text.strip()
        net_change = columns[2].text.strip()
        percent_change = columns[3].text.strip()
        month_change = columns[4].text.strip()
        year_change = columns[5].text.strip()
        time = columns[6].text.strip()

        # Добавляем данные в список
        data.append([name, value, net_change, percent_change, month_change, year_change, time])

# Записываем данные в CSV файл
with open('stock_data.csv', mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(headers)  # Записываем заголовки
    writer.writerows(data)  # Записываем строки данных

print("Данные успешно записаны в 'stock_data.csv'.")

# Закрываем браузер
driver.quit()
