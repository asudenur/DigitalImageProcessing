import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel,
                           QVBoxLayout, QWidget, QFileDialog, QHBoxLayout,
                           QFrame, QGridLayout, QStackedWidget, QTableWidget, QTableWidgetItem)
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt
import cv2
import numpy as np
import os
import pandas as pd

def display_cv_image(cv_img, label_widget):
    """Ortak görüntü gösterme fonksiyonu."""
    if cv_img is None:
        label_widget.setText("Görüntü yok.")
        label_widget.setPixmap(QPixmap())
        return
    if len(cv_img.shape) == 2:
        height, width = cv_img.shape
        bytes_per_line = width
        q_img = QImage(cv_img.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
    elif len(cv_img.shape) == 3:
        img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        height, width, channel = img_rgb.shape
        bytes_per_line = 3 * width
        q_img = QImage(img_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888) # OpenCV uses BGR
    else:
        label_widget.setText("Desteklenmeyen görüntü formatı.")
        return
    pixmap = QPixmap.fromImage(q_img)
    # scaled_pixmap = pixmap.scaled(label_widget.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation) # Etiket boyutu değişirse sorun olur
    label_widget.setPixmap(pixmap.scaled(label_widget.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

def load_image_dialog(parent, color_mode=cv2.IMREAD_COLOR):
    """Ortak dosya yükleme diyalog fonksiyonu."""
    file_path, _ = QFileDialog.getOpenFileName(parent, "Resim Seç", "", "Images (*.png *.jpg *.jpeg *.bmp)")
    if file_path:
        img = cv2.imread(file_path, color_mode)
        return img
    return None

class StyleSheet:
    MAIN_BACKGROUND = """
        QMainWindow {
            background-color: white;
        }
    """

    CONTAINER_STYLE = """
        QFrame {
            background-color: white;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 10px;
        }
    """

    TITLE_STYLE = """
        QLabel {
            color: black;
            font-weight: bold;
        }
    """

    BUTTON_STYLE = """
        QPushButton {
            background-color: white;
            color: black;
            border: 1px solid #ddd;
            padding: 8px;
            border-radius: 5px;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #f5f5f5;
        }
        QPushButton:pressed {
            background-color: #ebebeb;
        }
    """

    IMAGE_LABEL_STYLE = """
        QLabel {
            background-color: white;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 5px;
        }
    """

class FinalOdeviWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Final Ödevi - Uygulama Seçimi")
        self.setGeometry(100, 100, 900, 600)
        self.setStyleSheet(StyleSheet.MAIN_BACKGROUND)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        self.layout.setSpacing(20)
        self.layout.setContentsMargins(40, 40, 40, 40)

        # Başlık
        title_label = QLabel("Final Ödevi - Uygulama Seçimi")
        title_label.setFont(QFont("Arial", 20, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet(StyleSheet.TITLE_STYLE)
        self.layout.addWidget(title_label)

        # Stacked widget (sayfalar arası geçiş için)
        self.stacked_widget = QStackedWidget()
        self.layout.addWidget(self.stacked_widget, stretch=1)

        # Ana seçim sayfası
        self.selection_page = QWidget()
        selection_layout = QVBoxLayout(self.selection_page)
        selection_layout.setSpacing(30)
        selection_layout.setAlignment(Qt.AlignCenter)

        self.btn_scurve = QPushButton("1. S-Curve Kontrast Güçlendirme")
        self.btn_scurve.setStyleSheet(StyleSheet.BUTTON_STYLE)
        self.btn_scurve.setFont(QFont("Arial", 14))
        self.btn_scurve.setFixedSize(500, 60)
        self.btn_scurve.clicked.connect(self.show_scurve_page)
        selection_layout.addWidget(self.btn_scurve, alignment=Qt.AlignCenter)

        self.btn_hough = QPushButton("2. Hough Transform Uygulamaları")
        self.btn_hough.setStyleSheet(StyleSheet.BUTTON_STYLE)
        self.btn_hough.setFont(QFont("Arial", 14))
        self.btn_hough.setFixedSize(500, 60)
        self.btn_hough.clicked.connect(self.show_hough_page)
        selection_layout.addWidget(self.btn_hough, alignment=Qt.AlignCenter)

        self.btn_other1 = QPushButton("3. Deblurring (Motion Blur Düzeltme)")
        self.btn_other1.setStyleSheet(StyleSheet.BUTTON_STYLE)
        self.btn_other1.setFont(QFont("Arial", 14))
        self.btn_other1.setFixedSize(500, 60)
        self.btn_other1.setEnabled(True)
        self.btn_other1.clicked.connect(self.show_deblur_page)
        selection_layout.addWidget(self.btn_other1, alignment=Qt.AlignCenter)

        self.btn_other2 = QPushButton("4. Nesne Sayma ve Özellik Çıkarma")
        self.btn_other2.setStyleSheet(StyleSheet.BUTTON_STYLE)
        self.btn_other2.setFont(QFont("Arial", 14))
        self.btn_other2.setFixedSize(500, 60)
        self.btn_other2.setEnabled(True)
        self.btn_other2.clicked.connect(self.show_object_page)
        selection_layout.addWidget(self.btn_other2, alignment=Qt.AlignCenter)

        self.stacked_widget.addWidget(self.selection_page)

        # S-Curve sayfası
        self.scurve_page = SCurvePage(self)
        self.stacked_widget.addWidget(self.scurve_page)

        # Hough Transform sayfası (şimdilik şablon)
        self.hough_page = HoughPage(self)
        self.stacked_widget.addWidget(self.hough_page)

        # Deblurring sayfası
        self.deblur_page = DeblurPage(self)
        self.stacked_widget.addWidget(self.deblur_page)

        # Nesne Sayma ve Özellik Çıkarma sayfası
        self.object_page = ObjectFeaturePage(self)
        self.stacked_widget.addWidget(self.object_page)

        self.stacked_widget.setCurrentWidget(self.selection_page)

    def show_scurve_page(self):
        self.stacked_widget.setCurrentWidget(self.scurve_page)

    def show_hough_page(self):
        self.stacked_widget.setCurrentWidget(self.hough_page)

    def show_deblur_page(self):
        self.stacked_widget.setCurrentWidget(self.deblur_page)

    def show_object_page(self):
        self.stacked_widget.setCurrentWidget(self.object_page)

class SCurvePage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.original_image = None
        self.processed_image = None
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # Başlık ve geri dön
        top_layout = QHBoxLayout()
        back_btn = QPushButton("Geri Dön")
        back_btn.setStyleSheet(StyleSheet.BUTTON_STYLE)
        back_btn.setFont(QFont("Arial", 11))
        back_btn.setFixedWidth(120)
        back_btn.clicked.connect(self.go_back)
        top_layout.addWidget(back_btn)
        title = QLabel("S-Curve Kontrast Güçlendirme")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(StyleSheet.TITLE_STYLE)
        top_layout.addWidget(title, stretch=1)
        layout.addLayout(top_layout)

        # Kontrol Butonları
        controls_container = QFrame()
        controls_container.setStyleSheet(StyleSheet.CONTAINER_STYLE)
        controls_layout = QHBoxLayout(controls_container)
        self.load_button = QPushButton("Resim Yükle")
        self.load_button.setStyleSheet(StyleSheet.BUTTON_STYLE)
        self.load_button.setFixedSize(150, 40)
        self.load_button.setFont(QFont("Arial", 11))
        self.load_button.clicked.connect(self.load_image)
        controls_layout.addWidget(self.load_button)
        controls_layout.addStretch()
        layout.addWidget(controls_container)

        # Görüntü Alanları
        images_container = QFrame()
        images_container.setStyleSheet(StyleSheet.CONTAINER_STYLE)
        images_layout = QHBoxLayout(images_container)
        # Sol Görüntü
        input_image_frame = QFrame()
        input_image_layout = QVBoxLayout(input_image_frame)
        input_title = QLabel("Input Image")
        input_title.setFont(QFont("Arial", 14, QFont.Bold))
        input_title.setAlignment(Qt.AlignCenter)
        input_title.setStyleSheet(StyleSheet.TITLE_STYLE)
        self.input_image_label = QLabel("Resim yüklemek için butona tıklayın.")
        self.input_image_label.setAlignment(Qt.AlignCenter)
        self.input_image_label.setStyleSheet(StyleSheet.IMAGE_LABEL_STYLE)
        self.input_image_label.setMinimumSize(400, 300)
        input_image_layout.addWidget(input_title)
        input_image_layout.addWidget(self.input_image_label, stretch=1)
        images_layout.addWidget(input_image_frame, stretch=1)
        # Sağ Görüntü
        output_image_frame = QFrame()
        output_image_layout = QVBoxLayout(output_image_frame)
        output_title = QLabel("Final Output")
        output_title.setFont(QFont("Arial", 14, QFont.Bold))
        output_title.setAlignment(Qt.AlignCenter)
        output_title.setStyleSheet(StyleSheet.TITLE_STYLE)
        self.output_image_label = QLabel("İşlenmiş görüntü burada gösterilecek.")
        self.output_image_label.setAlignment(Qt.AlignCenter)
        self.output_image_label.setStyleSheet(StyleSheet.IMAGE_LABEL_STYLE)
        self.output_image_label.setMinimumSize(400, 300)
        output_image_layout.addWidget(output_title)
        output_image_layout.addWidget(self.output_image_label, stretch=1)
        images_layout.addWidget(output_image_frame, stretch=1)
        layout.addWidget(images_container, stretch=1)

        # Fonksiyon Butonları
        functions_container = QFrame()
        functions_container.setStyleSheet(StyleSheet.CONTAINER_STYLE)
        functions_layout = QGridLayout(functions_container)
        self.sigmoid_button = QPushButton("Standart Sigmoid")
        self.sigmoid_button.setStyleSheet(StyleSheet.BUTTON_STYLE)
        self.sigmoid_button.setFont(QFont("Arial", 11))
        self.sigmoid_button.clicked.connect(self.apply_standard_sigmoid)
        functions_layout.addWidget(self.sigmoid_button, 0, 0)
        self.shifted_sigmoid_button = QPushButton("Yatay Kaydırılmış Sigmoid")
        self.shifted_sigmoid_button.setStyleSheet(StyleSheet.BUTTON_STYLE)
        self.shifted_sigmoid_button.setFont(QFont("Arial", 11))
        self.shifted_sigmoid_button.clicked.connect(self.apply_shifted_sigmoid)
        functions_layout.addWidget(self.shifted_sigmoid_button, 0, 1)
        self.sloped_sigmoid_button = QPushButton("Eğimli Sigmoid")
        self.sloped_sigmoid_button.setStyleSheet(StyleSheet.BUTTON_STYLE)
        self.sloped_sigmoid_button.setFont(QFont("Arial", 11))
        self.sloped_sigmoid_button.clicked.connect(self.apply_sloped_sigmoid)
        functions_layout.addWidget(self.sloped_sigmoid_button, 1, 0)
        self.custom_function_button = QPushButton("Kendi Fonksiyonum")
        self.custom_function_button.setStyleSheet(StyleSheet.BUTTON_STYLE)
        self.custom_function_button.setFont(QFont("Arial", 11))
        self.custom_function_button.clicked.connect(self.apply_custom_function)
        functions_layout.addWidget(self.custom_function_button, 1, 1)
        self.reset_button = QPushButton("Sıfırla")
        self.reset_button.setStyleSheet(StyleSheet.BUTTON_STYLE)
        self.reset_button.setFont(QFont("Arial", 11))
        self.reset_button.clicked.connect(self.reset_output)
        functions_layout.addWidget(self.reset_button, 2, 0, 1, 2)
        layout.addWidget(functions_container)

    def go_back(self):
        self.parent.stacked_widget.setCurrentWidget(self.parent.selection_page)

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Resim Seç", "",
                                                 "Images (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            self.original_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if self.original_image is not None:
                display_cv_image(self.original_image, self.input_image_label)
                self.output_image_label.setText("İşlenmiş görüntü burada gösterilecek.")
                self.processed_image = None
            else:
                self.input_image_label.setText("Resim yüklenemedi.")

    def apply_standard_sigmoid(self):
        if self.original_image is None:
            self.output_image_label.setText("Önce bir resim yükleyin.")
            self.output_image_label.setPixmap(QPixmap())
            return
        img_norm = self.original_image.astype(np.float32) / 255.0
        gain = 10
        cutoff = 0.5
        processed_norm = 1 / (1 + np.exp(-gain * (img_norm - cutoff)))
        self.processed_image = (processed_norm * 255).astype(np.uint8)
        display_cv_image(self.processed_image, self.output_image_label)

    def apply_shifted_sigmoid(self):
        if self.original_image is None:
            self.output_image_label.setText("Önce bir resim yükleyin.")
            self.output_image_label.setPixmap(QPixmap())
            return
        img_norm = self.original_image.astype(np.float32) / 255.0
        gain = 10
        shift = 0.3
        processed_norm = 1 / (1 + np.exp(-gain * (img_norm - shift)))
        self.processed_image = (processed_norm * 255).astype(np.uint8)
        display_cv_image(self.processed_image, self.output_image_label)

    def apply_sloped_sigmoid(self):
        if self.original_image is None:
            self.output_image_label.setText("Önce bir resim yükleyin.")
            self.output_image_label.setPixmap(QPixmap())
            return
        img_norm = self.original_image.astype(np.float32) / 255.0
        slope = 20
        processed_norm = 1 / (1 + np.exp(-slope * (img_norm - 0.5)))
        self.processed_image = (processed_norm * 255).astype(np.uint8)
        display_cv_image(self.processed_image, self.output_image_label)

    def apply_custom_function(self):
        if self.original_image is None:
            self.output_image_label.setText("Önce bir resim yükleyin.")
            self.output_image_label.setPixmap(QPixmap())
            return
        img_norm = self.original_image.astype(np.float32) / 255.0
        processed_norm = 1 - (1 / (1 + np.exp(-10 * (img_norm - 0.5))))
        self.processed_image = (processed_norm * 255).astype(np.uint8)
        display_cv_image(self.processed_image, self.output_image_label)

    def reset_output(self):
        if self.original_image is not None:
            self.processed_image = None
            display_cv_image(self.original_image, self.output_image_label)

class HoughPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.original_image = None
        self.result_image = None
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        # Başlık ve geri dön
        top_layout = QHBoxLayout()
        back_btn = QPushButton("Geri Dön")
        back_btn.setStyleSheet(StyleSheet.BUTTON_STYLE)
        back_btn.setFont(QFont("Arial", 11))
        back_btn.setFixedWidth(120)
        back_btn.clicked.connect(self.go_back)
        top_layout.addWidget(back_btn)
        title = QLabel("Hough Transform Uygulamaları")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(StyleSheet.TITLE_STYLE)
        top_layout.addWidget(title, stretch=1)
        layout.addLayout(top_layout)

        # Butonlar
        btn_layout = QHBoxLayout()
        self.btn_load = QPushButton("Resim Yükle")
        self.btn_load.setStyleSheet(StyleSheet.BUTTON_STYLE)
        self.btn_load.setFont(QFont("Arial", 11))
        self.btn_load.clicked.connect(self.load_image)
        btn_layout.addWidget(self.btn_load)

        self.btn_lines = QPushButton("Yol Çizgisi Tespiti (Hough Line)")
        self.btn_lines.setStyleSheet(StyleSheet.BUTTON_STYLE)
        self.btn_lines.setFont(QFont("Arial", 11))
        self.btn_lines.clicked.connect(self.detect_lines)
        btn_layout.addWidget(self.btn_lines)

        self.btn_circles = QPushButton("Göz Tespiti (Hough Circle)")
        self.btn_circles.setStyleSheet(StyleSheet.BUTTON_STYLE)
        self.btn_circles.setFont(QFont("Arial", 11))
        self.btn_circles.clicked.connect(self.detect_circles)
        btn_layout.addWidget(self.btn_circles)

        layout.addLayout(btn_layout)

        # Görüntü alanı
        images_layout = QHBoxLayout()
        self.input_label = QLabel("Resim yükleyin.")
        self.input_label.setAlignment(Qt.AlignCenter)
        self.input_label.setStyleSheet(StyleSheet.IMAGE_LABEL_STYLE)
        self.input_label.setMinimumSize(350, 250)
        images_layout.addWidget(self.input_label)
        self.result_label = QLabel("Sonuç burada gösterilecek.")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet(StyleSheet.IMAGE_LABEL_STYLE)
        self.result_label.setMinimumSize(350, 250)
        images_layout.addWidget(self.result_label)
        layout.addLayout(images_layout)

    def go_back(self):
        self.parent.stacked_widget.setCurrentWidget(self.parent.selection_page)

    def load_image(self):
        self.original_image = load_image_dialog(self, cv2.IMREAD_COLOR)
        if self.original_image is not None:
            display_cv_image(self.original_image, self.input_label)
            self.result_label.setText("Sonuç burada gösterilecek.")
            self.result_label.setPixmap(QPixmap())
            self.result_image = None
        else:
            self.input_label.setText("Resim yüklenemedi.")

    def detect_lines(self):
        if self.original_image is None:
            self.result_label.setText("Önce bir resim yükleyin.")
            self.result_label.setPixmap(QPixmap())
            return
        img = self.original_image.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 110, apertureSize=3)

        # Kenar çizgileri için
        lines1 = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=35, minLineLength=90, maxLineGap=6)
        # Orta şeritler için
        lines2 = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=25, minLineLength=6, maxLineGap=7)

        if lines1 is not None:
            for line in lines1:
                x1, y1, x2, y2 = line[0]
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if lines2 is not None:
            for line in lines2:
                x1, y1, x2, y2 = line[0]
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        self.result_image = img
        display_cv_image(self.result_image, self.result_label)

    def detect_circles(self):
        if self.original_image is None:
            self.result_label.setText("Önce bir resim yükleyin.")
            self.result_label.setPixmap(QPixmap())
            return
        img = self.original_image.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        # Sadece üst yarıda arama
        roi = gray[0:int(h*0.6), :]
        roi = cv2.equalizeHist(roi)
        roi = cv2.medianBlur(roi, 5)
        circles = cv2.HoughCircles(
            roi, cv2.HOUGH_GRADIENT, dp=1.2, minDist=int(w*0.3),
            param1=50, param2=30, minRadius=12, maxRadius=30
        )
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                center = (i[0], i[1])
                cv2.circle(img, center, i[2], (0, 255, 0), 2)
                cv2.circle(img, center, 2, (0, 0, 255), 3)
        self.result_image = img
        display_cv_image(self.result_image, self.result_label)

class DeblurPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.original_image = None
        self.deblurred_image = None
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # Başlık ve geri dön
        top_layout = QHBoxLayout()
        back_btn = QPushButton("Geri Dön")
        back_btn.setStyleSheet(StyleSheet.BUTTON_STYLE)
        back_btn.setFont(QFont("Arial", 11))
        back_btn.setFixedWidth(120)
        back_btn.clicked.connect(self.go_back)
        top_layout.addWidget(back_btn)
        title = QLabel("Deblurring (Motion Blur Düzeltme)")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(StyleSheet.TITLE_STYLE)
        top_layout.addWidget(title, stretch=1)
        layout.addLayout(top_layout)

        # Kontrol Butonları
        controls_container = QFrame()
        controls_container.setStyleSheet(StyleSheet.CONTAINER_STYLE)
        controls_layout = QHBoxLayout(controls_container)
        self.load_button = QPushButton("Resim Yükle")
        self.load_button.setStyleSheet(StyleSheet.BUTTON_STYLE)
        self.load_button.setFixedSize(150, 40)
        self.load_button.setFont(QFont("Arial", 11))
        self.load_button.clicked.connect(self.load_image)
        controls_layout.addWidget(self.load_button)
        self.deblur_button = QPushButton("Deblur")
        self.deblur_button.setStyleSheet(StyleSheet.BUTTON_STYLE)
        self.deblur_button.setFixedSize(150, 40)
        self.deblur_button.setFont(QFont("Arial", 11))
        self.deblur_button.clicked.connect(self.apply_deblur)
        controls_layout.addWidget(self.deblur_button)
        controls_layout.addStretch()
        layout.addWidget(controls_container)

        # Görüntü Alanları
        images_container = QFrame()
        images_container.setStyleSheet(StyleSheet.CONTAINER_STYLE)
        images_layout = QHBoxLayout(images_container)
        # Sol Görüntü
        self.input_image_label = QLabel("Resim yüklemek için butona tıklayın.")
        self.input_image_label.setAlignment(Qt.AlignCenter)
        self.input_image_label.setStyleSheet(StyleSheet.IMAGE_LABEL_STYLE)
        self.input_image_label.setMinimumSize(400, 300)
        images_layout.addWidget(self.input_image_label, stretch=1)
        # Sağ Görüntü
        self.output_image_label = QLabel("Deblur sonucu burada gösterilecek.")
        self.output_image_label.setAlignment(Qt.AlignCenter)
        self.output_image_label.setStyleSheet(StyleSheet.IMAGE_LABEL_STYLE)
        self.output_image_label.setMinimumSize(400, 300)
        images_layout.addWidget(self.output_image_label, stretch=1)
        layout.addWidget(images_container, stretch=1)

    def go_back(self):
        self.parent.stacked_widget.setCurrentWidget(self.parent.selection_page)

    def load_image(self):
        self.original_image = load_image_dialog(self, cv2.IMREAD_COLOR)
        if self.original_image is not None:
            display_cv_image(self.original_image, self.input_image_label)
            self.output_image_label.setText("Deblur sonucu burada gösterilecek.")
            self.output_image_label.setPixmap(QPixmap())
            self.deblurred_image = None
        else:
            self.input_image_label.setText("Resim yüklenemedi.")

    def apply_deblur(self):
        if self.original_image is None:
            self.output_image_label.setText("Önce bir resim yükleyin.")
            self.output_image_label.setPixmap(QPixmap())
            return
        img = self.original_image
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # 1. Aşama: Renk canlandırma (CLAHE, daha güçlü)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(12,12))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        color_boosted = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        # 2. Aşama: HSV uzayında saturation artır
        hsv = cv2.cvtColor(color_boosted, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s = cv2.add(s, 30)
        s = np.clip(s, 0, 255).astype(np.uint8)
        hsv_boosted = cv2.merge((h, s, v))
        color_boosted2 = cv2.cvtColor(hsv_boosted, cv2.COLOR_HSV2BGR)

        # 3. Aşama: Güçlü unsharp masking (büyük kernel, yüksek ağırlık)
        gaussian1 = cv2.GaussianBlur(color_boosted2, (11, 11), 15.0)
        sharpened1 = cv2.addWeighted(color_boosted2, 2.0, gaussian1, -1.0, 0)

        # 4. Aşama: İkinci unsharp masking (daha küçük kernel, orta ağırlık)
        gaussian2 = cv2.GaussianBlur(sharpened1, (5, 5), 3.0)
        sharpened2 = cv2.addWeighted(sharpened1, 1.5, gaussian2, -0.5, 0)

        # 5. Aşama: Laplacian ile ekstra keskinlik
        laplacian = cv2.Laplacian(sharpened2, cv2.CV_64F)
        laplacian = np.clip(laplacian, 0, 255).astype(np.uint8)
        sharpened3 = cv2.addWeighted(sharpened2, 0.8, laplacian, 0.2, 0)

        # 6. Aşama: Sonuçları birleştir
        final = cv2.addWeighted(sharpened2, 0.5, sharpened3, 0.5, 0)

        self.deblurred_image = final
        display_cv_image(self.deblurred_image, self.output_image_label)

class ObjectFeaturePage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.original_image = None
        self.mask = None
        self.features = []  # Son bulunan özellikler burada tutulacak
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # Başlık ve geri dön
        top_layout = QHBoxLayout()
        back_btn = QPushButton("Geri Dön")
        back_btn.setStyleSheet(StyleSheet.BUTTON_STYLE)
        back_btn.setFont(QFont("Arial", 11))
        back_btn.setFixedWidth(120)
        back_btn.clicked.connect(self.go_back)
        top_layout.addWidget(back_btn)
        title = QLabel("Nesne Sayma ve Özellik Çıkarma")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(StyleSheet.TITLE_STYLE)
        top_layout.addWidget(title, stretch=1)
        layout.addLayout(top_layout)

        # Kontrol Butonları
        controls_container = QFrame()
        controls_container.setStyleSheet(StyleSheet.CONTAINER_STYLE)
        controls_layout = QHBoxLayout(controls_container)
        self.load_button = QPushButton("Görsel Yükle")
        self.load_button.setStyleSheet(StyleSheet.BUTTON_STYLE)
        self.load_button.setFixedSize(150, 40)
        self.load_button.setFont(QFont("Arial", 11))
        self.load_button.clicked.connect(self.load_image)
        controls_layout.addWidget(self.load_button)
        self.detect_button = QPushButton("Nesneleri Bul")
        self.detect_button.setStyleSheet(StyleSheet.BUTTON_STYLE)
        self.detect_button.setFixedSize(150, 40)
        self.detect_button.setFont(QFont("Arial", 11))
        self.detect_button.clicked.connect(self.detect_objects)
        controls_layout.addWidget(self.detect_button)
        self.export_button = QPushButton("Excel'e Aktar")
        self.export_button.setStyleSheet(StyleSheet.BUTTON_STYLE)
        self.export_button.setFixedSize(150, 40)
        self.export_button.setFont(QFont("Arial", 11))
        self.export_button.clicked.connect(self.export_to_excel)
        controls_layout.addWidget(self.export_button)
        controls_layout.addStretch()
        layout.addWidget(controls_container)

        # Görüntü Alanı
        images_container = QFrame()
        images_container.setStyleSheet(StyleSheet.CONTAINER_STYLE)
        images_layout = QHBoxLayout(images_container)
        self.input_image_label = QLabel("Görsel yükleyin.")
        self.input_image_label.setAlignment(Qt.AlignCenter)
        self.input_image_label.setStyleSheet(StyleSheet.IMAGE_LABEL_STYLE)
        self.input_image_label.setMinimumSize(400, 300)
        images_layout.addWidget(self.input_image_label, stretch=1)
        layout.addWidget(images_container, stretch=1)

        # Tablo Alanı
        self.table = QTableWidget()
        self.table.setColumnCount(9)
        self.table.setHorizontalHeaderLabels([
            "No", "Center", "Length", "Width", "Diagonal", "Energy", "Entropy", "Mean", "Median"
        ])
        layout.addWidget(self.table)

    def go_back(self):
        self.parent.stacked_widget.setCurrentWidget(self.parent.selection_page)

    def load_image(self):
        self.original_image = load_image_dialog(self, cv2.IMREAD_COLOR)
        if self.original_image is not None:
            display_cv_image(self.original_image, self.input_image_label)
            self.table.setRowCount(0)
        else:
            self.input_image_label.setText("Görsel yüklenemedi.")

    def detect_objects(self):
        if self.original_image is None:
            return
        img = self.original_image.copy()
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # Koyu yeşil için HSV aralığı (örnek)
        lower = np.array([35, 80, 20])
        upper = np.array([85, 255, 120])
        mask = cv2.inRange(hsv, lower, upper)
        self.mask = mask
        # Bağlantılı bileşen analizi
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        features = []
        for i in range(1, num_labels):  # 0 arka plan
            x, y, w, h, area = stats[i]
            cx, cy = centroids[i]
            roi = img[y:y+h, x:x+w]
            roi_mask = (labels[y:y+h, x:x+w] == i).astype(np.uint8)
            # Özellikler
            length = h
            width = w
            diagonal = int(np.sqrt(h**2 + w**2))
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            roi_pixels = roi_gray[roi_mask == 1]
            if roi_pixels.size == 0:
                continue
            energy = np.sum((roi_pixels/255.0)**2)
            hist = cv2.calcHist([roi_pixels], [0], None, [256], [0,256])
            hist_norm = hist / np.sum(hist)
            entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-8))
            mean = int(np.mean(roi_pixels))
            median = int(np.median(roi_pixels))
            features.append([
                i,
                f"{int(cx)},{int(cy)}",
                f"{length} px",
                f"{width} px",
                f"{diagonal} px",
                f"{energy:.3f}",
                f"{entropy:.2f}",
                mean,
                median
            ])
        self.features = features  # Excel için sakla
        self.table.setRowCount(len(features))
        for row, feat in enumerate(features):
            for col, val in enumerate(feat):
                self.table.setItem(row, col, QTableWidgetItem(str(val)))

    def export_to_excel(self):
        if not self.features:
            return
        df = pd.DataFrame(self.features, columns=[
            "No", "Center", "Length", "Width", "Diagonal", "Energy", "Entropy", "Mean", "Median"
        ])
        file_path, _ = QFileDialog.getSaveFileName(self, "Excel Dosyası Kaydet", "nesne_ozellikleri.xlsx", "Excel Files (*.xlsx)")
        if file_path:
            if not file_path.endswith('.xlsx'):
                file_path += '.xlsx'
            df.to_excel(file_path, index=False)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FinalOdeviWindow()
    window.show()
    sys.exit(app.exec_())
