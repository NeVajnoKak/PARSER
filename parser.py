import requests
from bs4 import BeautifulSoup
import csv

# URL страницы
url = "https://kolesa.kz/"

# Заголовки для имитации реального пользователя
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36"
}

# Загружаем страницу
response = requests.get(url, headers=headers)

# Проверяем успешность запроса
if response.status_code == 200:
    # Парсим HTML с помощью BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')

    # Ищем все теги <img> с атрибутом alt
    img_elements = soup.find_all('img', alt=True)

    # Открываем CSV файл для записи
    with open('output.csv', mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # Записываем заголовки столбцов
        writer.writerow(['Image URL', 'Description'])

        # Сохраняем данные в CSV
        for img in img_elements:
            img_src = img.get('src')  # Ссылка на изображение
            img_alt = img.get('alt')  # Описание изображения
            writer.writerow([img_src, img_alt])  # Записываем строку

    print("Данные успешно сохранены в 'output.csv'.")
else:
    print(f"Ошибка загрузки страницы: {response.status_code}")
