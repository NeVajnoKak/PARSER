# Пример проекта с анализом данных и парсингом с сайта Bloomberg

Этот проект демонстрирует использование различных методов машинного обучения для анализа данных и их отображения в графическом интерфейсе с использованием PyQt5. Включает в себя функции линейной регрессии, классификации и кластеризации, а также парсинг данных с сайта Bloomberg и сохранение результатов в CSV файл.

## Функциональность

1. **Линейная регрессия:** Прогнозирует цену квартиры на основе площади и этажа.
2. **Классификация:** Классифицирует квартиры по районам с использованием метода классификации.
3. **Кластеризация:** Кластеризует квартиры по площади и этажу с помощью алгоритма KMeans.
4. **Парсинг с Bloomberg:** Парсинг данных с сайта Bloomberg и сохранение их в CSV файл.

## Требования

- Python 3.6 и выше.
- Библиотеки:
  - PyQt5 для интерфейса.
  - Scikit-learn для алгоритмов машинного обучения.
  - Pandas для обработки данных.
  - Requests для работы с HTTP-запросами.
  - BeautifulSoup для парсинга HTML.

## Установка

Следуйте этим шагам, чтобы запустить проект локально:

### 1. Клонировать репозиторий

Сначала клонируйте репозиторий на свою машину:

```bash
git clone https://github.com/yourusername/yourproject.git
cd yourproject
```
### 2. Создание виртуального окружения
Создайте виртуальное окружение, чтобы изолировать зависимости проекта:

```bash
python3 -m venv venv
```

### 3. Активировать виртуальное окружение
Для активации виртуального окружения выполните следующую команду:

На macOS и Linux:
```bash
source venv/bin/activate
```
На Windows:
```bash
venv\Scripts\activate
```

После активации виртуального окружения, в командной строке появится префикс (venv).

### 4. Установка зависимостей
Установите все необходимые библиотеки с помощью pip:

```bash
pip install -r requirements.txt
```
### 5. Запуск проекта
После установки зависимостей, вы можете запустить проект с помощью следующей команды:

```bash
python main.py
```
Проект откроет графический интерфейс с кнопками для выполнения анализа данных с использованием линейной регрессии, классификации и кластеризации.

### Как работает парсинг данных с Bloomberg
Для парсинга данных с сайта Bloomberg мы используем библиотеку requests для получения HTML-страницы и BeautifulSoup для извлечения нужных данных.

1. Парсинг таблицы: Мы извлекаем таблицу с котировками из таблицы на странице https://www.bloomberg.com/markets/stocks/world-indexes/americas.
2. Сохранение в CSV: Все извлеченные данные сохраняются в CSV файл, чтобы их можно было обработать и проанализировать.
Чтобы запустить парсинг данных и сохранить их в CSV:

```bash
python parse_bloomberg.py
```
После этого файл parsed_apartments_data.csv будет сохранен в вашем проекте с результатами парсинга.

### Структура проекта
```bash
yourproject/
│
├── main.py                  # Основной файл с PyQt5 интерфейсом
├── parse_bloomberg.py       # Скрипт для парсинга данных с Bloomberg
├── parsed_apartments_data.csv # Результат парсинга в формате CSV
├── requirements.txt         # Список зависимостей проекта
├── venv/                    # Виртуальное окружение
├── README.md                # Документация проекта
```
## Использование
### Линейная регрессия
1. Введите параметры, такие как площадь и этаж.
2. Получите прогноз цены квартиры на основе данных.
## Классификация
Прогнозирует, к какому району относится квартира на основе площади и этажа.
## Кластеризация
Кластеризует квартиры в 3 группы по характеристикам (площадь и этаж).
## Парсинг с Bloomberg
1. Выполните парсинг таблицы с котировками с сайта Bloomberg.
2. Данные сохраняются в файл CSV.
## Зависимости
Чтобы установить зависимости, создайте виртуальное окружение и выполните:

```bash
pip install -r requirements.txt
```
Зависимости:

- pandas
- numpy
- scikit-learn
- requests
- beautifulsoup4
- PyQt5

## Примечания
- Для успешного парсинга данных с Bloomberg убедитесь, что сайт доступен, так как сайт может блокировать парсинг.
- Для использования PyQt5 на macOS потребуется Xcode для компиляции необходимых зависимостей.
## Лицензия
Этот проект лицензирован под лицензией MIT. Для получения дополнительной информации ознакомьтесь с файлом LICENSE.

- Автор: Erkebulan
- Дата: 2024
