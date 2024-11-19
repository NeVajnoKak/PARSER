import csv
import re


# Функция для парсинга информации из alt
def parse_alt(alt_text):
    # Пример alt: "3-комнатная квартира · 90 м² · 23/27 этаж, Ул. кошкарбаев 8 за 55 млн 〒 в Астане, Алматы р-н"

    # Регулярные выражения для извлечения данных
    rooms_pattern = r'(\d+)-комнатная квартира'
    area_pattern = r'(\d+\.?\d*) м²'
    floors_pattern = r'(\d+)/(\d+) этаж'
    address_pattern = r'Ул\.(.*?)за'  # Извлекаем улицу до 'за'
    price_pattern = r'за (\d+\.?\d*) млн'

    # Извлекаем данные с помощью регулярных выражений
    rooms = re.search(rooms_pattern, alt_text)
    area = re.search(area_pattern, alt_text)
    floors = re.search(floors_pattern, alt_text)
    address = re.search(address_pattern, alt_text)
    price = re.search(price_pattern, alt_text)

    data = {
        'rooms': int(rooms.group(1)) if rooms else None,
        'area': float(area.group(1)) if area else None,
        'floor': int(floors.group(1)) if floors else None,
        'total_floors': int(floors.group(2)) if floors else None,
        'address': address.group(1).strip() if address else None,
        'price': float(price.group(1)) if price else None,
    }

    return data


# Чтение CSV файла и обработка данных, а также сохранение в новый CSV
def read_and_save_parsed_data(input_filename, output_filename):
    parsed_data = []
    with open(input_filename, mode='r', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        next(reader)  # Пропускаем заголовки, если они есть
        for row in reader:
            alt_text = row[0]  # Предположим, что alt находится в первом столбце
            parsed_data.append(parse_alt(alt_text))

    # Сохраняем обработанные данные в новый CSV
    with open(output_filename, mode='w', newline='', encoding='utf-8') as outfile:
        fieldnames = ['rooms', 'area', 'floor', 'total_floors', 'address', 'price']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()  # Записываем заголовки
        for data in parsed_data:
            writer.writerow(data)

    print(f"Данные успешно сохранены в {output_filename}")


# Пример использования
input_filename = 'images_data.csv'  # Исходный файл с alt текстами
output_filename = 'parsed_apartments_data.csv'  # Новый файл для сохраненных данных
read_and_save_parsed_data(input_filename, output_filename)
