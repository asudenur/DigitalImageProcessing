import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel, 
                           QVBoxLayout, QWidget, QFileDialog, QHBoxLayout, 
                           QStackedWidget, QFrame, QSlider)
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt
import cv2
import numpy as np
import os
from odev2 import Odev2
from finalodevi import FinalOdeviWindow

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
    
    INFO_STYLE = """
        QLabel {
            color: black;
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
    
    RETURN_BUTTON_STYLE = """
        QPushButton {
            background-color: white;
            color: black;
            border: 1px solid #ddd;
            padding: 8px;
            border-radius: 5px;
        }
        QPushButton:hover {
            background-color: #f5f5f5;
        }
    """
    
    SLIDER_STYLE = """
        QSlider::groove:horizontal {
            border: none;
            height: 4px;
            background: #ddd;
            border-radius: 2px;
        }

        QSlider::handle:horizontal {
            background: white;
            border: 1px solid #ddd;
            width: 16px;
            height: 16px;
            margin: -6px 0;
            border-radius: 8px;
        }
        
        QSlider::handle:horizontal:hover {
            background: #f5f5f5;
        }
    """
    
    LABEL_STYLE = """
        QLabel {
            color: black;
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

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dijital Görüntü İşleme")
        self.setGeometry(100, 100, 1450, 850)
        self.setStyleSheet(StyleSheet.MAIN_BACKGROUND)
        
        # Ana widget ve düzen
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # Yığın widget oluşturma
        self.stacked_widget = QStackedWidget()
        
        # Ana sayfa
        self.main_page = QWidget()
        self.create_main_page()
        self.stacked_widget.addWidget(self.main_page)
        
        # Ödev 1 sayfası
        self.homework1_page = QWidget()
        self.create_homework1_page()
        self.stacked_widget.addWidget(self.homework1_page)
        
        # Ana düzen
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.stacked_widget)
        self.central_widget.setLayout(main_layout)
        
        # Ana sayfayı göster
        self.stacked_widget.setCurrentWidget(self.main_page)
        
        # Değişkenler
        self.image_path = None
        self.current_image = None
        self.original_image = None
        
        # Değişkenler for Final Ödevi
        self.final_original_image = None
        self.final_processed_image = None
        
    def create_main_page(self):
        layout = QVBoxLayout()
        layout.setSpacing(20)
        layout.setContentsMargins(40, 40, 40, 40)
        
        # Başlık container'ı
        title_container = QFrame()
        title_container.setStyleSheet(StyleSheet.CONTAINER_STYLE)
        title_layout = QVBoxLayout()
        
        # Başlık
        title = QLabel("DİJİTAL GÖRÜNTÜ İŞLEME")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Arial", 24))
        title.setStyleSheet(StyleSheet.TITLE_STYLE)
        title_layout.addWidget(title)
        
        # Öğrenci bilgileri
        info = QLabel("Öğrenci No: 221229013\nAd Soyad: Asude Nur Demir")
        info.setAlignment(Qt.AlignCenter)
        info.setFont(QFont("Arial", 14))
        info.setStyleSheet(StyleSheet.INFO_STYLE)
        title_layout.addWidget(info)
        
        title_container.setLayout(title_layout)
        layout.addWidget(title_container)
        
        # Ödev butonu container'ı
        button_container = QFrame()
        button_container.setStyleSheet(StyleSheet.CONTAINER_STYLE)
        button_layout = QVBoxLayout()
        
        # Ödev 1 butonu
        homework1_button = QPushButton("Ödev 1: Temel İşlevsellik")
        homework1_button.setStyleSheet(StyleSheet.BUTTON_STYLE)
        homework1_button.setFixedSize(400, 50)
        homework1_button.setFont(QFont("Arial", 14))
        homework1_button.clicked.connect(lambda: self.stacked_widget.setCurrentWidget(self.homework1_page))
        button_layout.addWidget(homework1_button, alignment=Qt.AlignCenter)
        
        # Ödev 2 butonu
        homework2_button = QPushButton("Ödev 2: Temel Görüntü Operasyonları ve İnterpolasyon ")
        homework2_button.setStyleSheet(StyleSheet.BUTTON_STYLE)
        homework2_button.setFixedSize(660, 50)
        homework2_button.setFont(QFont("Arial", 14))
        homework2_button.clicked.connect(self.open_odev2)
        button_layout.addWidget(homework2_button, alignment=Qt.AlignCenter)
        
        # Final Ödevi butonu
        final_homework_button = QPushButton("Final Ödevi")
        final_homework_button.setStyleSheet(StyleSheet.BUTTON_STYLE)
        final_homework_button.setFixedSize(660, 50)
        final_homework_button.setFont(QFont("Arial", 14))
        final_homework_button.clicked.connect(self.open_final_odevi)
        button_layout.addWidget(final_homework_button, alignment=Qt.AlignCenter)
        
        button_container.setLayout(button_layout)
        layout.addWidget(button_container)
        
        layout.addStretch()
        self.main_page.setLayout(layout)
        
    def create_homework1_page(self):
        layout = QVBoxLayout()
        layout.setSpacing(20)
        layout.setContentsMargins(40, 40, 40, 40)
        
        # Üst kısım container'ı
        top_container = QFrame()
        top_container.setStyleSheet(StyleSheet.CONTAINER_STYLE)
        top_layout = QHBoxLayout()
        
        # Geri dönüş butonu
        return_button = QPushButton("Ana Sayfa")
        return_button.setStyleSheet(StyleSheet.RETURN_BUTTON_STYLE)
        return_button.setFixedSize(120, 35)
        return_button.setFont(QFont("Arial", 11))
        return_button.clicked.connect(lambda: self.stacked_widget.setCurrentWidget(self.main_page))
        top_layout.addWidget(return_button)
        
        # Başlık
        title = QLabel("Ödev 1: Temel İşlevsellik")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Arial", 16))
        title.setStyleSheet(StyleSheet.TITLE_STYLE)
        top_layout.addWidget(title)
        
        # Sağ tarafta boş widget (simetri için)
        empty_widget = QWidget()
        empty_widget.setFixedSize(120, 35)
        top_layout.addWidget(empty_widget)
        
        top_container.setLayout(top_layout)
        layout.addWidget(top_container)
        
        # İşlevler container'ı
        content_container = QFrame()
        content_container.setStyleSheet(StyleSheet.CONTAINER_STYLE)
        content_layout = QVBoxLayout()
        
        # Kontrol butonları
        button_layout = QHBoxLayout()
        
        # Dosya işlemleri
        self.load_button = QPushButton("Resim Yükle")
        self.save_button = QPushButton("Kaydet")
        self.reset_button = QPushButton("Sıfırla")
        
        for button in [self.load_button, self.save_button, self.reset_button]:
            button.setStyleSheet(StyleSheet.BUTTON_STYLE)
            button.setFixedSize(150, 40)
            button.setFont(QFont("Arial", 11))
            button_layout.addWidget(button)
            
        button_layout.addStretch()
        
        # Görüntü işleme
        self.gray_button = QPushButton("Gri Tonlama")
        self.gray_button.setStyleSheet(StyleSheet.BUTTON_STYLE)
        self.gray_button.setFixedSize(150, 40)
        self.gray_button.setFont(QFont("Arial", 11))
        button_layout.addWidget(self.gray_button)
        
        content_layout.addLayout(button_layout)
        
        # Ayarlar container'ı
        settings_container = QFrame()
        settings_container.setStyleSheet(StyleSheet.CONTAINER_STYLE)
        settings_layout = QVBoxLayout()
        
        # Parlaklık ayarı
        brightness_layout = QHBoxLayout()
        brightness_label = QLabel("Parlaklık")
        brightness_label.setStyleSheet(StyleSheet.LABEL_STYLE)
        brightness_label.setFont(QFont("Arial", 11))
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setStyleSheet(StyleSheet.SLIDER_STYLE)
        self.brightness_slider.setRange(-50, 50)
        self.brightness_slider.setValue(0)
        brightness_layout.addWidget(brightness_label)
        brightness_layout.addWidget(self.brightness_slider)
        settings_layout.addLayout(brightness_layout)
        
        # Kontrast ayarı
        contrast_layout = QHBoxLayout()
        contrast_label = QLabel("Kontrast")
        contrast_label.setStyleSheet(StyleSheet.LABEL_STYLE)
        contrast_label.setFont(QFont("Arial", 11))
        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setStyleSheet(StyleSheet.SLIDER_STYLE)
        self.contrast_slider.setRange(0, 200)
        self.contrast_slider.setValue(100)
        contrast_layout.addWidget(contrast_label)
        contrast_layout.addWidget(self.contrast_slider)
        settings_layout.addLayout(contrast_layout)
        
        settings_container.setLayout(settings_layout)
        content_layout.addWidget(settings_container)
        
        # Görüntü alanı
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet(StyleSheet.IMAGE_LABEL_STYLE)
        self.image_label.setMinimumSize(800, 400)
        content_layout.addWidget(self.image_label)
        
        content_container.setLayout(content_layout)
        layout.addWidget(content_container)
        
        self.homework1_page.setLayout(layout)
        
        # Bağlantılar
        self.setup_connections()
        
    def setup_connections(self):
        self.load_button.clicked.connect(self.load_image)
        self.save_button.clicked.connect(self.save_image)
        self.reset_button.clicked.connect(self.reset_image)
        self.gray_button.clicked.connect(self.convert_to_gray)
        self.brightness_slider.valueChanged.connect(self.adjust_brightness_contrast)
        self.contrast_slider.valueChanged.connect(self.adjust_brightness_contrast)
        
    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Resim Seç", "", 
                                                 "Images (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            self.image_path = file_path
            self.original_image = cv2.imread(file_path)
            self.current_image = self.original_image.copy()
            self.update_image_display()
            
    def save_image(self):
        if self.current_image is not None:
            file_path, _ = QFileDialog.getSaveFileName(self, "Görüntüyü Kaydet", "", 
                                                     "PNG (*.png);;JPEG (*.jpg *.jpeg);;BMP (*.bmp)")
            if file_path:
                if not any(file_path.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.bmp']):
                    file_path += '.png'
                cv2.imwrite(file_path, self.current_image)
                
    def reset_image(self):
        if self.original_image is not None:
            self.current_image = self.original_image.copy()
            self.brightness_slider.setValue(0)
            self.contrast_slider.setValue(100)
            self.update_image_display()
            
    def convert_to_gray(self):
        if self.current_image is not None:
            self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
            if len(self.current_image.shape) == 2:
                self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_GRAY2BGR)
            self.update_image_display()
            
    def adjust_brightness_contrast(self):
        if self.original_image is not None:
            brightness = self.brightness_slider.value()
            contrast = self.contrast_slider.value() / 100.0
            
            self.current_image = cv2.convertScaleAbs(
                self.original_image,
                alpha=contrast,
                beta=brightness
            )
            self.update_image_display()
            
    def update_image_display(self):
        if self.current_image is not None:
            # Ensure current_image is BGR before converting to RGB for display
            img_to_display = self.current_image
            if len(img_to_display.shape) == 2: # If it's grayscale
                img_to_display = cv2.cvtColor(img_to_display, cv2.COLOR_GRAY2BGR)
            
            height, width = img_to_display.shape[:2]
            bytes_per_line = 3 * width # BGR image
            
            image = cv2.cvtColor(img_to_display, cv2.COLOR_BGR2RGB)
            q_img = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            
            pixmap = QPixmap.fromImage(q_img)
            # Scale to image_label's size, which might be different from final page's labels
            scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_label.setPixmap(scaled_pixmap)

    def open_odev2(self):
        self.odev2_window = Odev2()
        self.odev2_window.show()

    def open_final_odevi(self):
        self.final_window = FinalOdeviWindow()
        self.final_window.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())