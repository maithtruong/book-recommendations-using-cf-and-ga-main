import typing
from PyQt5.QtWidgets import QApplication,QMainWindow, QTableWidgetItem, QLabel, QHBoxLayout, QWidget, QTableWidget, QListWidgetItem, QMessageBox
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap, QMovie
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from GD import Ui_MainWindow
from data_import import import_user_ratings, import_semantic
from bliga import BookRecommendationSystem
import csv

class BookRecommendationThread(QThread):
    finished = pyqtSignal(object)

    def __init__(self, user_ratings_input, user_ratings, semantic, pop_size, mutation_rate, num_generations, crossover_func, mutation_func, no_rec):
        QThread.__init__(self)
        self.user_ratings_input = user_ratings_input
        self.user_ratings = user_ratings
        self.semantic = semantic
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.num_generations = num_generations
        self.crossover_func = crossover_func
        self.mutation_func = mutation_func
        self.no_rec = no_rec

    def run(self):
        recommender = BookRecommendationSystem(self.user_ratings_input, self.user_ratings, self.semantic)
        best_solution = recommender.genetic_algorithm(
            self.pop_size, self.mutation_rate, self.num_generations, self.crossover_func, self.mutation_func, self.no_rec
        )
        self.finished.emit(best_solution)
class Loading(QWidget):
    def __init__(self,mainwindow):
        super().__init__(mainwindow)
        
        self.verticalLayout = QtWidgets.QVBoxLayout(self)
        self.label_ani = QLabel(self)
        self.movie = QMovie("load.gif")
        self.label_ani.setMovie(self.movie)
        self.setFixedSize(170,150)
        self.verticalLayout.addWidget(self.label_ani, 0, QtCore.Qt.AlignHCenter)
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        self.startani()
        
    def showEvent(self, event):
        main_window_rect = self.parent().geometry()
        loading_rect = self.geometry()
        center_x = int(main_window_rect.center().x() - loading_rect.width() / 2)
        center_y = int(main_window_rect.center().y() - loading_rect.height() / 2)
        self.move(center_x, center_y)

    def startani(self):
        self.movie.start()

    def stopani(self):
        self.movie.stop()
        self.close()

         
class StarRating(QWidget):
    def __init__(self, star_count=5):
        super().__init__()

        self.star_count = star_count
        self.stars = []

        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        for i in range(star_count):
            star = QLabel(self)
            pixmap = QPixmap('star_unselected.png')
            star.setPixmap(pixmap)
            star.setScaledContents(True)

            star.setAlignment(Qt.AlignCenter)
            self.layout.addWidget(star)
            self.stars.append(star)

    def mousePressEvent(self, event):
        for i in range(self.star_count):
            if self.stars[i].underMouse():
                self.setRating(i + 1)
                break

    def setRating(self, rating):
        
        for i in range(self.star_count):
            if i < rating:
                self.stars[i].setPixmap(QPixmap('star_selected.png'))  
            else:
                self.stars[i].setPixmap(QPixmap('star_unselected.png'))  

class UI():
    def __init__(self):
        self.mainUI = QMainWindow()
        self.main = Ui_MainWindow()
        self.main.setupUi(self.mainUI)
        self.mainUI.show()
        
        self.display_books("BOOKS.csv")
        self.first_click = True
        self.main.search_input.textChanged.connect(self.search_perform)
        self.main.search_results_list.doubleClicked.connect(self.select_book)
        self.book_titles = []
        self.main.bt_submit.clicked.connect(self.get_input_data)
        
        self.main.input_userdata.setColumnCount(7) 
        self.main.input_userdata.setHorizontalHeaderItem(0, QTableWidgetItem("ID"))
        self.main.input_userdata.setHorizontalHeaderItem(1, QTableWidgetItem("Title"))
        self.main.input_userdata.setHorizontalHeaderItem(2, QTableWidgetItem("Author"))
        self.main.input_userdata.setHorizontalHeaderItem(3, QTableWidgetItem("Page Count"))
        self.main.input_userdata.setHorizontalHeaderItem(4, QTableWidgetItem("Genres"))
        self.main.input_userdata.setHorizontalHeaderItem(5, QTableWidgetItem("Language"))
        self.main.input_userdata.setHorizontalHeaderItem(6, QTableWidgetItem("Published Date"))

        
        self.main.input_userdata.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)  
        self.main.search_results_list.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.main.input_userdata.verticalHeader().setVisible(False)

        
        self.main.input_userdata.setColumnWidth(0, 30)
        self.main.input_userdata.setColumnWidth(1, 400)
        self.main.input_userdata.setColumnWidth(2, 130)
        self.main.input_userdata.setColumnWidth(3, 80)
        self.main.input_userdata.setColumnWidth(4, 180)
        self.main.input_userdata.setColumnWidth(5, 70)
        self.main.input_userdata.setColumnWidth(6, 100)
        

    def display_books(self, file_path):
        with open(file_path, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            header = next(csvreader)
            self.main.input_userdata.setColumnCount(len(header) - 2)  

            for row_num, row_data in enumerate(csvreader):
                self.main.input_userdata.insertRow(row_num)
                col_count = 0
                for col_num, value in enumerate(row_data):
                    if col_num not in [3,4,5]:
                        item = QTableWidgetItem(value)
                        self.main.input_userdata.setItem(row_num, col_count, item)
                        col_count += 1


    def search_perform(self,text):
        if text.strip() == '':
            self.main.search_results_list.hide()
        else:
            search_results = self.get_related_search_results(text)
            model = QtCore.QStringListModel()
            model.setStringList(search_results)
            self.main.search_results_list.setModel(model)
            self.main.search_results_list.raise_()
            self.main.search_results_list.show()

    def get_related_search_results(self, search_text):
        related_search_terms = self.read_titles_from_csv('BOOKS.csv')
        filtered_results = [term for term in related_search_terms if search_text.lower() in term.lower()]
        return filtered_results
    
    def read_titles_from_csv(self,file_path):
        titles = []
        with open(file_path, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                title = row['Title']
                titles.append(title)
        return titles
    
    def select_book(self, index):
        selected_book = index.data()
        if selected_book not in self.book_titles:
            self.book_titles.append(selected_book)
            if self.first_click:
                self.main.input_userdata.clear()
                self.main.input_userdata.setRowCount(0)
                self.main.input_userdata.setColumnCount(3)
                self.main.input_userdata.setHorizontalHeaderItem(0, QTableWidgetItem("Book ID"))
                self.main.input_userdata.setHorizontalHeaderItem(1, QTableWidgetItem("Book's Title"))
                self.main.input_userdata.setHorizontalHeaderItem(2, QTableWidgetItem("Your Rating"))
                self.main.input_userdata.setColumnWidth(0, 80)
                self.main.input_userdata.setColumnWidth(1, 730)
                self.main.input_userdata.setColumnWidth(2, 220)
                self.main.input_userdata.verticalHeader().setVisible(False)
                self.first_click = False
            book_id = self.get_book_id(selected_book)
            self.add_to_user_inputdata(book_id, selected_book)
            
    def get_book_id(self,book_title):
        with open('BOOKS.csv', 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            header = next(csvreader)
            for row_data in csvreader:
                if row_data[1] == book_title:
                    return row_data[0]
        return None


    def add_to_user_inputdata(self,book_id, book_title):
        row_count = self.main.input_userdata.rowCount()
        self.main.input_userdata.setRowCount(row_count + 1)

        self.main.input_userdata.setItem(row_count, 1, QTableWidgetItem(book_title))

        rating_widget = StarRating()
        self.main.input_userdata.setCellWidget(row_count, 2, rating_widget)

        item = QTableWidgetItem(str(book_id))
        self.main.input_userdata.setItem(row_count, 0, item)
        
    def get_input_data(self):
        if self.main.input_userdata.rowCount() < 5:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("You need to select at least 5 books!")
            msg.setWindowTitle("Warning")
            msg.exec_()
            return
        input_data = []
        total = self.read_titles_from_csv("BOOKS.csv")
        for i in range(len(total)):
            input_data.append(0)
        row_count = self.main.input_userdata.rowCount()
        
        for row in range(row_count):
            book_id_item = self.main.input_userdata.item(row, 0)
            rating_widget = self.main.input_userdata.cellWidget(row, 2)
            
            if book_id_item and rating_widget:
                book_id = int(book_id_item.text())
                rating = rating_widget.star_count
                
                input_data[book_id - 1] = rating
        
        user_ratings_file = "user_ratings.csv"
        semantic_file = "semantic.csv"

        user_ratings = import_user_ratings(user_ratings_file)
        semantic = import_semantic(semantic_file)
        user_ratings_input = input_data
        
        pop_size = 50   
        mutation_rate = 0.1
        num_generations = 5
        crossover_func = "one_point"
        mutation_func = "swap"
        no_rec = 10
        
        self.loading = Loading(self.mainUI)
        self.loading.startani()
        self.loading.show()
        self.recommendation_thread = BookRecommendationThread(user_ratings_input, user_ratings, semantic, pop_size, mutation_rate, num_generations, crossover_func, mutation_func, no_rec)
        self.recommendation_thread.finished.connect(self.on_recommendation_finished)
        self.recommendation_thread.start()

    def get_book_titles(self, book_id):
        with open('BOOKS.csv', 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            header = next(csvreader)
            for row_data in csvreader:
                if row_data[0] == book_id:
                    return row_data[1]
        return None

    
    def on_recommendation_finished(self, best_solution):
        self.loading.stopani()
        self.main.output_bookrcm.setColumnCount(2)
        self.main.output_bookrcm.setHorizontalHeaderItem(0, QTableWidgetItem("Book ID"))
        self.main.output_bookrcm.setHorizontalHeaderItem(1, QTableWidgetItem("Book's Title"))
        self.main.output_bookrcm.setColumnWidth(1, 900)
        self.main.output_bookrcm.verticalHeader().setVisible(False)
        recommended_books = best_solution  
        for book in recommended_books:
            titles = self.get_book_titles(str(book+1))
            row = self.main.output_bookrcm.rowCount()
            self.main.output_bookrcm.insertRow(row)
            self.main.output_bookrcm.setItem(row,0,QTableWidgetItem(str(book+1)))
            self.main.output_bookrcm.setItem(row,1,QTableWidgetItem(str(titles)))
            print(book)

        
                
        
if __name__ == "__main__":
    app = QApplication([])
    ui = UI()
    app.exec_()
