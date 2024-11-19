import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QMessageBox
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Пример: Лин. Регр., Классификация, Кластеризация")
        self.setGeometry(200, 200, 400, 300)

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

        # Настраиваем центральный виджет
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def linear_regression_action(self):
        description = (
            "Линейная регрессия используется для поиска зависимости между переменными.\n\n"
            "Инструкция:\n"
            "1. Определите переменные X (входные данные) и y (целевая переменная).\n"
            "2. Используйте алгоритм для обучения модели.\n\n"
            "Пример задачи: Найти зависимость зарплаты от опыта."
        )
        QMessageBox.information(self, "Линейная Регрессия", description)

        # Пример 1: Зависимость зарплаты от опыта
        X = np.array([[1], [2], [3], [4], [5]])  # Опыт в годах
        y = np.array([40, 50, 60, 70, 80])  # Зарплата в тысячах $

        model = LinearRegression()
        model.fit(X, y)
        prediction = model.predict([[6]])  # Предсказать зарплату для 6 лет опыта

        QMessageBox.information(
            self,
            "Пример 1",
            f"Опыт 6 лет -> Зарплата: {prediction[0]:.2f} тыс. $",
        )

        # Пример 2: Зависимость продаж от рекламы
        X = np.array([[10], [20], [30], [40], [50]])  # Затраты на рекламу (тыс. $)
        y = np.array([15, 25, 35, 50, 60])  # Продажи (тыс. $)

        model.fit(X, y)
        prediction = model.predict([[60]])  # Предсказать продажи для затрат 60 тыс. $

        QMessageBox.information(
            self,
            "Пример 2",
            f"Реклама 60 тыс. $ -> Продажи: {prediction[0]:.2f} тыс. $",
        )

    def classification_action(self):
        description = (
            "Классификация используется для разделения объектов на категории.\n\n"
            "Инструкция:\n"
            "1. Соберите данные с метками (классы: 0, 1 и т. д.).\n"
            "2. Обучите модель классификации.\n\n"
            "Пример задачи: Разделить точки на классы."
        )
        QMessageBox.information(self, "Классификация", description)

        # Пример 1: Классификация точек
        X, y = make_classification(n_samples=20, n_features=2, n_classes=2, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = DecisionTreeClassifier()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        QMessageBox.information(
            self,
            "Пример 1",
            f"Тестовые данные: {X_test[0]} -> Предсказанный класс: {predictions[0]}",
        )

        # Пример 2: Прогноз успеваемости
        features = np.array([[85, 80], [70, 65], [90, 95], [60, 50]])  # Баллы (тесты, проекты)
        labels = np.array([1, 0, 1, 0])  # 1 - успешен, 0 - неуспешен

        model.fit(features, labels)
        prediction = model.predict([[75, 85]])  # Прогноз для нового ученика

        QMessageBox.information(
            self,
            "Пример 2",
            f"Баллы: 75, 85 -> Успех: {prediction[0]}",
        )

    def clustering_action(self):
        description = (
            "Кластеризация используется для группировки похожих объектов.\n\n"
            "Инструкция:\n"
            "1. Подготовьте данные без меток.\n"
            "2. Используйте алгоритм кластеризации для разделения на группы.\n\n"
            "Пример задачи: Группировать точки на 2 кластера."
        )
        QMessageBox.information(self, "Кластеризация", description)

        # Пример 1: Группировка точек
        X = np.array([[1, 2], [2, 3], [10, 10], [12, 12], [5, 5], [6, 6]])  # Пример данных
        model = KMeans(n_clusters=2, random_state=42)
        model.fit(X)
        labels = model.labels_

        QMessageBox.information(
            self,
            "Пример 1",
            f"Точки: {X.tolist()} -> Кластеры: {labels.tolist()}",
        )

        # Пример 2: Группировка клиентов
        clients = np.array([[100, 200], [150, 250], [300, 500], [400, 600]])  # Покупки и расходы
        model = KMeans(n_clusters=2, random_state=42)
        model.fit(clients)
        labels = model.labels_

        QMessageBox.information(
            self,
            "Пример 2",
            f"Клиенты: {clients.tolist()} -> Кластеры: {labels.tolist()}",
        )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
