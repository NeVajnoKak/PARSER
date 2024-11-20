import pandas as pd
import re

# Загрузка CSV
input_file = "output.csv"
output_file = "processed_output.csv"

# Функция для разбора описания
def parse_description(description):
    if pd.isna(description) or 'года за' not in description:
        return ["", "", "", "", ""]

    # Разделяем описание с помощью регулярных выражений
    pattern = r"(.+?) (\d{4}) года за ([\d\s]+)(тг\.?) в (.+)"
    match = re.match(pattern, description)
    if match:
        model = match.group(1).strip()
        year = match.group(2).strip()
        price = match.group(3).replace(" ", "").strip()
        currency = match.group(4).strip()
        city = match.group(5).strip()
        return [model, year, price, currency, city]
    return ["", "", "", "", ""]

# Загрузка данных
df = pd.read_csv(input_file)

# Создание новых колонок
df[['Модель', 'Год', 'Цена', 'Валюта', 'Город']] = df['Description'].apply(parse_description).tolist()

# Удаляем старую колонку Description
df = df.drop(columns=['Description'])

# Сохранение результата
df.to_csv(output_file, index=False, encoding="utf-8")

print(f"Данные успешно обработаны и сохранены в {output_file}")
