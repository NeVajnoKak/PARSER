import sys
import csv
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QMessageBox, QTableWidget, QTableWidgetItem, QHBoxLayout
from PyQt5.QtCore import Qt
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Пример: Лин. Регр., Классификация, Кластеризация")
        self.setGeometry(200, 200, 800, 600)

        # Создаем виджеты
        layout = QVBoxLayout()

        self.btn_regression = QPushButton("Линейная Регрессия")
        self.btn_regression.clicked.connect(self.linear_regression_action)
        layout.addWidget(self.btn_regression)

        self.btn_classification = QPushButton("Классификация")
        self.btn_classification.clicked.connect(self.classification_action)
        layout.addWidget(self.btn_classification)

        self.btn_clustering = QPushButton("Кластеризация")
        self.btn_clustering.clicked.connect(self.clustering_action)
        layout.addWidget(self.btn_clustering)

        self.btn_exit = QPushButton("Выход")
        self.btn_exit.clicked.connect(self.close)
        layout.addWidget(self.btn_exit)

        # Добавляем область для отображения таблицы и графика
        self.table_widget = QTableWidget()
        layout.addWidget(self.table_widget)

        self.figure = plt.figure(figsize=(5, 3))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # Настраиваем центральный виджет
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Данные из CSV файла
        self.data = self.load_data_from_csv("parsed_apartments_data.csv")

    def load_data_from_csv(self, filename):
        """Загрузка данных из CSV в pandas DataFrame"""
        return pd.read_csv(filename)

    def linear_regression_action(self):
        description = (
            "Линейная регрессия используется для поиска зависимости между переменными.\n\n"
            "Инструкция:\n"
            "1. Определите переменные X (входные данные) и y (целевая переменная).\n"
            "2. Используйте алгоритм для обучения модели.\n\n"
            "Пример задачи: Найти зависимость цены квартиры от площади и этажа."
        )
        QMessageBox.information(self, "Линейная Регрессия", description)

        # Извлечение данных из CSV для линейной регрессии
        data = self.data.dropna(subset=['area', 'floor', 'price'])  # Убираем строки с пропущенными значениями
        X = data[['area', 'floor']].values  # Используем площадь и этаж
        y = data['price'].values  # Целевая переменная — цена

        model = LinearRegression()
        model.fit(X, y)
        prediction = model.predict([[75, 5]])  # Прогнозируем цену для квартиры площадью 75 м² на 5 этаже

        QMessageBox.information(
            self,
            "Пример 1",
            f"Цена квартиры 75 м² на 5 этаже: {prediction[0]:.2f} млн. ₸",
        )

        # Отображение таблицы с данными
        self.show_table(data)

        # Построение графика линейной регрессии
        self.plot_linear_regression(X, y, model)

    def show_table(self, data):
        """Отображение данных в таблице"""
        self.table_widget.setRowCount(data.shape[0])
        self.table_widget.setColumnCount(data.shape[1])

        # Устанавливаем заголовки
        self.table_widget.setHorizontalHeaderLabels(data.columns)

        # Заполняем таблицу данными
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                self.table_widget.setItem(i, j, QTableWidgetItem(str(data.iloc[i, j])))

    def plot_linear_regression(self, X, y, model):
        """Построение графика линейной регрессии"""
        plt.clf()  # Очистить предыдущие графики
        plt.scatter(X[:, 0], y, color='blue', label='Данные')  # Точки данных
        plt.plot(X[:, 0], model.predict(X), color='red', label='Регрессия')  # Линия регрессии
        plt.xlabel('Площадь')
        plt.ylabel('Цена')
        plt.title('Линейная Регрессия: Площадь vs. Цена')
        plt.legend()
        self.canvas.draw()

    def classification_action(self):
        description = (
            "Классификация используется для разделения объектов на категории.\n\n"
            "Инструкция:\n"
            "1. Соберите данные с метками (классы: 0, 1 и т. д.).\n"
            "2. Обучите модель классификации.\n\n"
            "Пример задачи: Классифицировать квартиры по районам."
        )
        QMessageBox.information(self, "Классификация", description)

        # Пример классификации по району
        data = self.data.dropna(subset=['price', 'area'])  # Убираем строки с пропущенными значениями
        X = data[['area', 'floor']].values  # Признаки: площадь и этаж
        y = data['address'].astype('category').cat.codes  # Метки: адрес/район квартиры (преобразуем в числа)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = DecisionTreeClassifier()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        QMessageBox.information(
            self,
            "Пример 1",
            f"Предсказанный район: {predictions[0]}",
        )

        # Отображение таблицы с результатами классификации
        result = pd.DataFrame({'Feature': ['area', 'floor'], 'Prediction': predictions.tolist()})
        self.show_table(result)

        # График классификации
        self.plot_classification(X, y, model)

    def plot_classification(self, X, y, model):
        """Построение графика классификации"""
        plt.clf()  # Очистить предыдущие графики
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
        plt.title('График классификации')
        plt.xlabel('Площадь')
        plt.ylabel('Этаж')
        self.canvas.draw()

    def clustering_action(self):
        description = (
            "Кластеризация используется для группировки похожих объектов.\n\n"
            "Инструкция:\n"
            "1. Подготовьте данные без меток.\n"
            "2. Используйте алгоритм кластеризации для разделения на группы.\n\n"
            "Пример задачи: Кластеризация квартир по характеристикам."
        )
        QMessageBox.information(self, "Кластеризация", description)

        # Пример кластеризации квартир по площади и этажности
        data = self.data.dropna(subset=['area', 'floor'])  # Убираем строки с пропущенными значениями
        X = data[['area', 'floor']].values  # Признаки: площадь и этаж

        model = KMeans(n_clusters=3, random_state=42)
        model.fit(X)
        labels = model.labels_

        QMessageBox.information(
            self,
            "Пример 1",
            f"Кластеры: {labels.tolist()}",
        )

        # Отображение таблицы с результатами кластеризации
        result = pd.DataFrame({'Cluster': labels})
        self.show_table(result)

        # График кластеризации
        self.plot_clustering(X, labels)

    def plot_clustering(self, X, labels):
        """Построение графика кластеризации"""
        plt.clf()  # Очистить предыдущие графики
        plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
        plt.title('График кластеризации')
        plt.xlabel('Площадь')
        plt.ylabel('Этаж')
        self.canvas.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
