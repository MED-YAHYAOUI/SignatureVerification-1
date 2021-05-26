import os
import sys
from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5 import QtCore, QtGui, QtWidgets
from keras import load_model

button_style = """QWidget { border: 1px solid #000; }
QPushButton { background-color: rgb(255, 255, 255);
              color: rgb(0, 0, 0);
              border: 1px solid rgb(0, 0, 0); }
QPushButton::hover { background-color: rgb(0, 0, 0);
                     color: rgb(255, 255, 255); }
QPushButton::pressed { background-color: rgba(220, 138, 255, 0.5);
                       color: rgb(0, 0, 0); }"""


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.model = load_model("")

        self.setWindowTitle("Signature Forgery Detection")
        self.resize(800, 600)
        self.setMinimumSize(QtCore.QSize(800, 600))

        font = QtGui.QFont()

        # central widget ######################################################################
        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.main_layout = QtWidgets.QGridLayout(self.centralwidget)
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        self.main_layout.setSpacing(5)

        # heading ######################################################################
        self.heading = QtWidgets.QLabel(self.centralwidget)
        self.heading.setMinimumSize(QtCore.QSize(0, 100))
        self.heading.setText("Signature Forgery Detection")
        font.setFamily("SimSun")
        font.setPointSize(32)
        font.setBold(True)
        font.setWeight(75)
        self.heading.setFont(font)
        self.heading.setAlignment(QtCore.Qt.AlignCenter)
        self.heading.setWordWrap(True)
        self.main_layout.addWidget(self.heading, 0, 0, 1, 1)

        # byline ######################################################################
        self.byline = QtWidgets.QLabel(self.centralwidget)
        self.byline.setMinimumSize(QtCore.QSize(0, 50))
        self.byline.setText("By: Ishani Kathuria")
        font.setFamily("SimSun")
        font.setPointSize(20)
        self.byline.setFont(font)
        self.byline.setAlignment(QtCore.Qt.AlignCenter)
        self.main_layout.addWidget(self.byline, 1, 0, 1, 1)

        # inner widget ######################################################################
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setMinimumSize(QtCore.QSize(0, 350))
        self.widget.setStyleSheet(button_style)
        self.image_layout = QtWidgets.QGridLayout(self.widget)
        self.image_layout.setContentsMargins(10, 10, 10, 10)
        self.image_layout.setSpacing(5)

        # original image ######################################################################
        self.original = QtWidgets.QLabel(self.widget)
        self.original.setMinimumSize(QtCore.QSize(230, 230))
        self.original.setText("TextLabel")
        self.original.setScaledContents(True)
        self.original.setAlignment(QtCore.Qt.AlignCenter)
        self.image_layout.addWidget(self.original, 0, 0, 2, 1)

        # upload original button #####
        self.upload_original = QtWidgets.QPushButton(self.widget)
        self.upload_original.setMinimumSize(QtCore.QSize(230, 50))
        self.upload_original.setText("Upload original image")
        font.setFamily("Courier")
        font.setPointSize(8)
        self.upload_original.setFont(font)
        self.upload_original.clicked.connect(self.click_upload)
        self.image_layout.addWidget(self.upload_original, 2, 0, 1, 1)

        # answer to - is it a forgery? ######################################################################
        self.answer = QtWidgets.QLabel(self.widget)
        self.answer.setMinimumSize(QtCore.QSize(150, 150))
        self.answer.setText("Loading...")
        font.setFamily("OCR A Extended")
        font.setPointSize(16)
        self.answer.setFont(font)
        self.answer.setAlignment(QtCore.Qt.AlignCenter)
        self.answer.setWordWrap(True)
        self.image_layout.addWidget(self.answer, 0, 1, 1, 1)

        # detected text ######################################################################
        self.detection = QtWidgets.QLabel(self.widget)
        self.detection.setMinimumSize(QtCore.QSize(230, 60))
        self.detection.setText("<p>Text Detected:</p><p>AaBbYyZz</p>")
        font.setFamily("OCR A Extended")
        font.setPointSize(12)
        self.detection.setFont(font)
        self.detection.setAlignment(QtCore.Qt.AlignCenter)
        self.image_layout.addWidget(self.detection, 1, 1, 1, 1)

        # forgery image ######################################################################
        self.forgery = QtWidgets.QLabel(self.widget)
        self.forgery.setMinimumSize(QtCore.QSize(230, 230))
        self.forgery.setText("TextLabel")
        self.forgery.setScaledContents(True)
        self.forgery.setAlignment(QtCore.Qt.AlignCenter)
        self.image_layout.addWidget(self.forgery, 0, 2, 2, 1)

        # upload forgery button #####
        self.upload_forgery = QtWidgets.QPushButton(self.widget)
        self.upload_forgery.setMinimumSize(QtCore.QSize(230, 50))
        self.upload_forgery.setText("Upload image to be checked")
        font.setFamily("Courier")
        font.setPointSize(8)
        self.upload_forgery.setFont(font)
        self.upload_forgery.clicked.connect(self.click_upload)
        self.image_layout.addWidget(self.upload_forgery, 2, 2, 1, 1)

        self.main_layout.addWidget(self.widget, 2, 0, 1, 1)
        self.setCentralWidget(self.centralwidget)

        # menubar ######################################################################
        self.menubar = QtWidgets.QMenuBar(self)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 25))
        self.about = QtWidgets.QMenu(self.menubar)
        self.about.setTitle("About")
        self.setMenuBar(self.menubar)
        self.menubar.addAction(self.about.menuAction())

        # statusbar ######################################################################
        self.statusbar = QtWidgets.QStatusBar(self)
        self.setStatusBar(self.statusbar)
    
    def click_upload(self):
        sender = self.sender().text()

        home_dir = str(os.getcwd())
        img = QtWidgets.QFileDialog.getOpenFileName(None, 'Open File', home_dir)[0]

        if sender == "Upload original image":
            self.original.setPixmap(QtGui.QPixmap(img))
        elif sender == "Upload image to be checked":
            self.forgery.setPixmap(QtGui.QPixmap(img))
        else:
            self.answer.setText("Invalid response.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()

    app.exec()
