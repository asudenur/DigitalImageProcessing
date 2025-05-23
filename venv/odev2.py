import cv2
import numpy as np
import math
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel, 
                           QFileDialog, QHBoxLayout, QFrame, QSlider, QInputDialog, QMessageBox,
                           QDialog, QVBoxLayout, QRadioButton, QButtonGroup)
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import Qt

# --- _cubic_kernel fonksiyonunu buraya ekleyin ---
def _cubic_kernel(x, a=-0.5):
    """Keys' cubic convolution kernel (a=-0.5 for Catmull-Rom)."""
    x = abs(float(x)) # Mutlak değer ve float'a çevirme
    x2 = x * x
    x3 = x2 * x
    if x < 1.0:
        return (a + 2.0) * x3 - (a + 3.0) * x2 + 1.0
    elif 1.0 <= x < 2.0:
        return a * x3 - 5.0 * a * x2 + 8.0 * a * x - 4.0 * a
    else:
        return 0.0
# -------------------------------------------------

class InterpolationDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("İnterpolasyon Yöntemi Seçin")
        self.setModal(True)
        
        layout = QVBoxLayout()
        
        # Radio butonları oluştur
        self.bilinear_radio = QRadioButton("Bilinear")
        self.bicubic_radio = QRadioButton("Bicubic")
        self.average_radio = QRadioButton("Average")
        
        # Varsayılan seçimi ayarla
        self.bilinear_radio.setChecked(True)
        
        # Butonları layout'a ekle
        layout.addWidget(self.bilinear_radio)
        layout.addWidget(self.bicubic_radio)
        layout.addWidget(self.average_radio)
        # Onay butonları
        buttons_layout = QHBoxLayout()
        ok_button = QPushButton("Tamam")
        cancel_button = QPushButton("İptal")
        
        ok_button.clicked.connect(self.accept)
        cancel_button.clicked.connect(self.reject)
        
        buttons_layout.addWidget(ok_button)
        buttons_layout.addWidget(cancel_button)
        layout.addLayout(buttons_layout)
        
        self.setLayout(layout)
    
    def get_selected_method(self):
        if self.bilinear_radio.isChecked():
            return "bilinear"
        elif self.bicubic_radio.isChecked():
            return "bicubic"

        else:
            return "average"

class Odev2(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('Ödev 2')
        self.setGeometry(100, 100, 1200, 800)
        
        layout = QVBoxLayout()
        layout.setSpacing(20)
        layout.setContentsMargins(40, 40, 40, 40)
        
        # Üst başlık
        top_container = QFrame()
        top_container.setStyleSheet ("""
            QFrame {
                background-color: white;
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 10px;
            }
        """)
        top_layout = QHBoxLayout()
        title = QLabel("Ödev 2: Temel Görüntü Operasyonları ve İnterpolasyon")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Arial", 16))
        title.setStyleSheet("color: black; font-weight: bold;")
        top_layout.addWidget(title)
        top_container.setLayout(top_layout)
        layout.addWidget(top_container)
        
        # İçerik
        content_container = QFrame()
        content_container.setStyleSheet("""
            QFrame {
                background-color: white;
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 10px;
            }
        """)
        content_layout = QVBoxLayout()
        
        # Kontrol butonları
        button_layout = QHBoxLayout()
        self.load_button = QPushButton("Resim Yükle")
        self.save_button = QPushButton("Kaydet")
        self.reset_button = QPushButton("Sıfırla")
        for button in [self.load_button, self.save_button, self.reset_button]:
            button.setStyleSheet("""
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
            """)
            button.setFixedSize(130, 40)
            button.setFont(QFont("Arial", 11))
            button_layout.addWidget(button)
        button_layout.addStretch()
        content_layout.addLayout(button_layout)
        
        # Ödev 2 fonksiyon butonları
        op_layout = QHBoxLayout()
        self.btn_enlarge = QPushButton("Büyüt")
        self.btn_shrink = QPushButton("Küçült")
        self.btn_zoom = QPushButton("Zoom In/Out")
        self.btn_rotate = QPushButton("Döndür")
        for btn in [self.btn_enlarge, self.btn_shrink, self.btn_zoom, self.btn_rotate]:
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #e6f2ff;
                    color: #003366;
                    border: 1px solid #99ccff;
                    padding: 8px;
                    border-radius: 5px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #cce6ff;
                }
            """)
            btn.setFixedSize(160, 40)
            btn.setFont(QFont("Arial", 11))
            op_layout.addWidget(btn)
        op_layout.addStretch()
        content_layout.addLayout(op_layout)
        
        # Görüntü alanı
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("""
            QLabel {
                background-color: white;
                border: 1px solid #ddd;
                border-radius: 5px;
            }
        """)
        self.image_label.setMinimumSize(800, 400)
        content_layout.addWidget(self.image_label)
        
        content_container.setLayout(content_layout)
        layout.addWidget(content_container)
        self.setLayout(layout)
        
        # Bağlantılar
        self.setup_connections()
        
        # Değişkenler
        self.image_path = None
        self.current_image = None
        self.original_image = None
        
    def setup_connections(self):
        self.load_button.clicked.connect(self.load_image)
        self.save_button.clicked.connect(self.save_image)
        self.reset_button.clicked.connect(self.reset_image)
        self.btn_enlarge.clicked.connect(self.enlarge_image)
        self.btn_shrink.clicked.connect(self.shrink_image)
        self.btn_zoom.clicked.connect(self.zoom_image)
        self.btn_rotate.clicked.connect(self.rotate_image)
        
    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Resim Seç", "", 
                                             "Images (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            self.image_path = file_path
            self.original_image = cv2.imread(file_path)
            if self.original_image is None:
                QMessageBox.critical(self, "Hata", "Resim yüklenemedi. Lütfen dosya yolunu ve formatını kontrol edin.")
                return
            self.current_image = self.original_image.copy()
            self.update_image_display()
        
    def save_image(self):
        if self.current_image is not None:
            file_path, _ = QFileDialog.getSaveFileName(self, "Görüntüyü Kaydet", "", 
                                                      "PNG (*.png);;JPEG (*.jpg *.jpeg);;BMP (*.bmp)")
            if file_path:
                if not any(file_path.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.bmp']):
                    file_path += '.png'
                try:
                    # Convert current_image to numpy array if it's a list
                    if isinstance(self.current_image, list):
                        img_array = np.array(self.current_image, dtype=np.uint8)
                    else:
                        img_array = self.current_image
                    
                    # Save the image
                    cv2.imwrite(file_path, img_array)
                    QMessageBox.information(self, "Başarılı", "Görüntü başarıyla kaydedildi!")
                except Exception as e:
                    QMessageBox.critical(self, "Hata", f"Görüntü kaydedilirken bir hata oluştu: {str(e)}")
        
    def reset_image(self):
        if self.original_image is not None:
            self.current_image = self.original_image.copy()
            self.update_image_display()
        
    def update_image_display(self):
        if self.current_image is not None:
            img = self.current_image
            # Eğer saf Python listesi ise numpy array'e çevir
            if isinstance(img, list):
                img = np.array(img, dtype=np.uint8)
            # Eğer tek kanal ise 3 kanala çevir
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            height, width = img.shape[:2]
            q_img = QImage(img.data, width, height, 3 * width, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            # Orijinal boyutta göster
            self.image_label.setPixmap(pixmap)
            # Label'ın boyutunu görüntünün boyutuna göre ayarla
            self.image_label.setFixedSize(width, height)

    def _get_interpolation_method_and_image_array(self, dialog_title_operation_name):
        """Handles image check, interpolation dialog, and image array preparation."""
        if self.current_image is None:
            QMessageBox.information(self, "Bilgi", "Lütfen önce bir resim yükleyin.")
            return None, None, None # img_arr, method, display_name

        dialog = InterpolationDialog(self)
        dialog.setWindowTitle(f"{dialog_title_operation_name} İnterpolasyon Yöntemi")
        user_selected_method = "bilinear"  # Default if dialog is cancelled early

        if dialog.exec_() == QDialog.Accepted:
            user_selected_method = dialog.get_selected_method()
        else:
            return None, None, None # User cancelled

        display_method_name = user_selected_method.capitalize()
        QMessageBox.information(self, "Bilgi", f"{display_method_name} interpolasyonu ile {dialog_title_operation_name.lower()} yapılacak.")

        if isinstance(self.current_image, list):
            img_arr = np.array(self.current_image, dtype=np.uint8)
        else:
            img_arr = self.current_image.copy()
        
        return img_arr, user_selected_method, display_method_name

    def _perform_manual_image_transformation(self, input_img_arr, new_width, new_height, get_source_coords_lambda, interpolation_method):
        """Core manual image transformation loop using a lambda for source coordinates."""
        orig_height, orig_width = input_img_arr.shape[:2]
        channels = input_img_arr.shape[2] if len(input_img_arr.shape) == 3 else 1

        if channels == 1:
            output_img = np.zeros((new_height, new_width), dtype=np.uint8)
        else:
            output_img = np.zeros((new_height, new_width, channels), dtype=np.uint8)

        for y_out in range(new_height):
            for x_out in range(new_width):
                x_src, y_src = get_source_coords_lambda(x_out, y_out)
                
                interpolated_pixel = self._get_interpolated_pixel(input_img_arr, x_src, y_src, interpolation_method)
                
                if interpolated_pixel is not None:
                    if channels == 1:
                        output_img[y_out, x_out] = interpolated_pixel
                    else:
                        output_img[y_out, x_out, :] = interpolated_pixel
        return output_img

    def enlarge_image(self):
        factor, ok = QInputDialog.getDouble(self, "Büyütme Faktörü", "Büyütme oranı (örn: 2):", 2, 1.01, 10, 2)
        if not ok:
            return

        img_arr, user_selected_method, display_method_name = self._get_interpolation_method_and_image_array("Büyütme")
        if img_arr is None:
            return

        orig_height, orig_width = img_arr.shape[:2]
        new_height = int(orig_height * factor)
        new_width = int(orig_width * factor)

        get_source_coords = lambda x_out, y_out: (x_out / factor, y_out / factor)
        
        enlarged_img = self._perform_manual_image_transformation(img_arr, new_width, new_height, get_source_coords, user_selected_method)
        
        self.current_image = enlarged_img
        self.update_image_display()
        QMessageBox.information(self, "Bilgi", f"Büyütme işlemi ({display_method_name}) tamamlandı. Yeni boyut: {new_width}x{new_height}")

    def shrink_image(self):
        factor, ok = QInputDialog.getDouble(self, "Küçültme Faktörü", "Küçültme oranı (0.1-0.99):", 0.5, 0.1, 0.99, 2)
        if not ok:
            return

        img_arr, user_selected_method, display_method_name = self._get_interpolation_method_and_image_array("Küçültme")
        if img_arr is None:
            return

        orig_height, orig_width = img_arr.shape[:2]
        new_height = int(orig_height * factor)
        new_width = int(orig_width * factor)
        new_height = max(1, new_height)
        new_width = max(1, new_width)

        get_source_coords = lambda x_out, y_out: (x_out / factor, y_out / factor)
        
        shrunken_img = self._perform_manual_image_transformation(img_arr, new_width, new_height, get_source_coords, user_selected_method)
        
        self.current_image = shrunken_img
        self.update_image_display()
        QMessageBox.information(self, "Bilgi", f"Küçültme işlemi ({display_method_name}) tamamlandı. Yeni boyut: {new_width}x{new_height}")

    def rotate_image(self):
        angle_deg, ok = QInputDialog.getDouble(self, "Döndürme Açısı", "Açı (derece):", 45, -360, 360, 1)
        if not ok:
            return

        img_arr, user_selected_method, display_method_name = self._get_interpolation_method_and_image_array("Döndürme")
        if img_arr is None:
            return
        
        orig_height, orig_width = img_arr.shape[:2]
        angle_rad = math.radians(angle_deg)
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
        center_x_orig, center_y_orig = orig_width / 2.0, orig_height / 2.0

        corners = [(-center_x_orig, -center_y_orig), (orig_width-center_x_orig, -center_y_orig), 
                   (-center_x_orig, orig_height-center_y_orig), (orig_width-center_x_orig, orig_height-center_y_orig)]
        rotated_corners = [(x*cos_a - y*sin_a, x*sin_a + y*cos_a) for x, y in corners]
        min_x, max_x = min(c[0] for c in rotated_corners), max(c[0] for c in rotated_corners)
        min_y, max_y = min(c[1] for c in rotated_corners), max(c[1] for c in rotated_corners)
        new_width, new_height = int(math.ceil(max_x - min_x)), int(math.ceil(max_y - min_y))
        new_center_x, new_center_y = new_width / 2.0, new_height / 2.0

        def get_source_coords_rotate(x_out_new, y_out_new):
            x_d, y_d = x_out_new - new_center_x, y_out_new - new_center_y
            x_s_center_rel = x_d * cos_a + y_d * sin_a
            y_s_center_rel = -x_d * sin_a + y_d * cos_a
            return x_s_center_rel + center_x_orig, y_s_center_rel + center_y_orig
        
        rotated_img = self._perform_manual_image_transformation(img_arr, new_width, new_height, get_source_coords_rotate, user_selected_method)
        
        self.current_image = rotated_img
        self.update_image_display()
        QMessageBox.information(self, "Bilgi", f"Döndürme işlemi ({display_method_name}) {angle_deg} derece tamamlandı. Yeni boyut: {new_width}x{new_height}")

    def zoom_image(self):
        zoom_type, ok_type = QInputDialog.getItem(self, "Zoom Seçimi", "Zoom yönünü seçin:", ["Zoom In", "Zoom Out"], 0, False)
        if not ok_type:
            return
        
        factor, ok_factor = QInputDialog.getDouble(self, "Zoom Faktörü", "Zoom oranı (örn: 1.5):", 1.5, 1.1, 5, 2)
        if not ok_factor:
            return

        img_arr, user_selected_method, display_method_name = self._get_interpolation_method_and_image_array(f"{zoom_type}")
        if img_arr is None:
            return
        
        orig_height, orig_width = img_arr.shape[:2]
        channels = img_arr.shape[2] if len(img_arr.shape) == 3 else 1
        final_image = None

        if zoom_type == "Zoom In":
            # Adım 1: Manuel büyütme
            temp_height_zoomed = int(orig_height * factor)
            temp_width_zoomed = int(orig_width * factor)
            get_source_coords_step1 = lambda x_out, y_out: (x_out / factor, y_out / factor)
            zoomed_img_step1 = self._perform_manual_image_transformation(img_arr, temp_width_zoomed, temp_height_zoomed, get_source_coords_step1, user_selected_method)

            # Adım 2: Kırpma
            center_y_zoomed, center_x_zoomed = temp_height_zoomed // 2, temp_width_zoomed // 2
            start_y_crop = max(0, center_y_zoomed - orig_height // 2)
            start_x_crop = max(0, center_x_zoomed - orig_width // 2)
            end_y_crop = min(temp_height_zoomed, start_y_crop + orig_height)
            end_x_crop = min(temp_width_zoomed, start_x_crop + orig_width)
            cropped_img_from_zoomed = zoomed_img_step1[start_y_crop:end_y_crop, start_x_crop:end_x_crop]
            cropped_height, cropped_width = cropped_img_from_zoomed.shape[:2]

            if cropped_width == 0 or cropped_height == 0:
                QMessageBox.warning(self, "Uyarı", "Zoom In sonrası kırpılan alan boş, işlem iptal edildi.")
                return

            # Adım 3: Kırpılmışı orijinal boyuta manuel "fit" etme
            # (Bu effectively bir resize işlemi, source coords hesaplaması farklı)
            if channels == 1:
                final_image_buffer = np.zeros((orig_height, orig_width), dtype=np.uint8)
            else:
                final_image_buffer = np.zeros((orig_height, orig_width, channels), dtype=np.uint8)
            
            scale_x_final = float(cropped_width) / orig_width # Kaynak (cropped) / Hedef (original)
            scale_y_final = float(cropped_height) / orig_height
            
            get_source_coords_step3 = lambda x_out, y_out: (x_out * scale_x_final, y_out * scale_y_final)
            final_image = self._perform_manual_image_transformation(cropped_img_from_zoomed, orig_width, orig_height, get_source_coords_step3, user_selected_method)

        else: # Zoom Out
            # Adım 1: Manuel küçültme
            temp_height_shrunk = int(orig_height / factor)
            temp_width_shrunk = int(orig_width / factor)
            temp_height_shrunk = max(1, temp_height_shrunk)
            temp_width_shrunk = max(1, temp_width_shrunk)
            
            # Küçültme için kaynak koordinat hesaplaması: x_src = x_out / (1/factor) = x_out * factor
            get_source_coords_shrink = lambda x_out, y_out: (x_out * factor, y_out * factor)
            shrunken_img = self._perform_manual_image_transformation(img_arr, temp_width_shrunk, temp_height_shrunk, get_source_coords_shrink, user_selected_method)
            
            # Adım 2: Tuvale yerleştirme
            if channels == 1:
                final_image = np.zeros((orig_height, orig_width), dtype=np.uint8)
            else:
                final_image = np.zeros((orig_height, orig_width, channels), dtype=np.uint8)
            
            start_y_place = max(0, (orig_height - temp_height_shrunk) // 2)
            start_x_place = max(0, (orig_width - temp_width_shrunk) // 2)
            place_h = min(temp_height_shrunk, orig_height - start_y_place)
            place_w = min(temp_width_shrunk, orig_width - start_x_place)
            
            if place_h > 0 and place_w > 0 and shrunken_img.shape[0] >= place_h and shrunken_img.shape[1] >= place_w :
                final_image[start_y_place : start_y_place + place_h, start_x_place : start_x_place + place_w] = \
                    shrunken_img[0:place_h, 0:place_w]
            else:
                # Eğer shrunken_img beklenenden küçükse veya place_h/w negatifse, bu bir hatadır, sadece siyah tuval kalsın
                QMessageBox.warning(self, "Uyarı", "Zoom Out sırasında boyutlandırma sorunu, görüntü boş olabilir.")
                # final_image zaten siyah bir tuval olduğu için ek bir şey yapmaya gerek yok

        if final_image is not None:
            self.current_image = final_image
            self.update_image_display()
            QMessageBox.information(self, "Bilgi", 
                                 f"{zoom_type} işlemi ({display_method_name}) tamamlandı. Yeni boyut: {final_image.shape[1]}x{final_image.shape[0]}")
        else:
            QMessageBox.warning(self, "Hata", "Zoom işlemi sırasında bir sorun oluştu.")

    # İnterpolasyon için yardımcı metod (isteğe bağlı, kod tekrarını azaltır)
    def _get_interpolated_pixel(self, img, x_src, y_src, method):
        orig_height, orig_width = img.shape[:2]
        channels = img.shape[2] if len(img.shape) == 3 else 1
        pixel_value = None # Başlangıçta None

        try:
            if method == "bilinear":
                if 0 <= x_src < orig_width - 1 and 0 <= y_src < orig_height - 1:
                    x1, y1 = int(math.floor(x_src)), int(math.floor(y_src))
                    x2, y2 = x1 + 1, y1 + 1
                    dx, dy = x_src - x1, y_src - y1
                    
                    if channels == 1:
                        p1,p2,p3,p4 = img[y1,x1], img[y1,x2], img[y2,x1], img[y2,x2]
                        val = (p1*(1-dx)*(1-dy) + p2*dx*(1-dy) + p3*(1-dx)*dy + p4*dx*dy)
                        pixel_value = np.clip(val, 0, 255).astype(np.uint8)
                    else:
                        final_val = [0.0]*channels
                        for ch in range(channels):
                            p1,p2,p3,p4 = float(img[y1,x1,ch]), float(img[y1,x2,ch]), float(img[y2,x1,ch]), float(img[y2,x2,ch])
                            val = (p1*(1-dx)*(1-dy) + p2*dx*(1-dy) + p3*(1-dx)*dy + p4*dx*dy)
                            final_val[ch] = val
                        pixel_value = np.clip(np.array(final_val), 0, 255).astype(np.uint8)

            elif method == "average": # 2x2 Box Average
                x1, y1 = int(math.floor(x_src)), int(math.floor(y_src))
                # Check if the 2x2 box is fully inside
                if 0 <= x1 < orig_width - 1 and 0 <= y1 < orig_height - 1:
                    x2, y2 = x1 + 1, y1 + 1
                    if channels == 1:
                        p1,p2,p3,p4 = float(img[y1,x1]), float(img[y1,x2]), float(img[y2,x1]), float(img[y2,x2])
                        avg_val = (p1 + p2 + p3 + p4) / 4.0
                        pixel_value = np.clip(avg_val, 0, 255).astype(np.uint8)
                    else:
                        final_val = [0.0]*channels
                        for ch in range(channels):
                            p1,p2,p3,p4 = float(img[y1,x1,ch]), float(img[y1,x2,ch]), float(img[y2,x1,ch]), float(img[y2,x2,ch])
                            avg_val = (p1 + p2 + p3 + p4) / 4.0
                            final_val[ch] = avg_val
                        pixel_value = np.clip(np.array(final_val), 0, 255).astype(np.uint8)
                # Else: Box is partly outside, return None (black pixel)

            elif method == "bicubic":
                x_fl, y_fl = int(math.floor(x_src)), int(math.floor(y_src))
                dx, dy = x_src - x_fl, y_src - y_fl
                
                accum_val = [0.0] * channels if channels > 1 else 0.0
                valid_contribution = False # Check if any pixel contributed

                for j in range(-1, 3): # y neighborhood index: -1, 0, 1, 2
                    y_n = y_fl + j
                    if 0 <= y_n < orig_height:
                        wy = _cubic_kernel(dy - j) # Use the helper function
                        for i in range(-1, 3): # x neighborhood index: -1, 0, 1, 2
                            x_n = x_fl + i
                            if 0 <= x_n < orig_width:
                                wx = _cubic_kernel(dx - i)
                                weight = wx * wy
                                neighbor_pixel = img[y_n, x_n]
                                valid_contribution = True # At least one pixel is valid
                                
                                if channels == 1:
                                    accum_val += float(neighbor_pixel) * weight
                                else:
                                    for ch in range(channels):
                                        accum_val[ch] += float(neighbor_pixel[ch]) * weight
                
                if valid_contribution:
                    if channels == 1:
                        pixel_value = np.clip(accum_val, 0, 255).astype(np.uint8)
                    else:
                        pixel_value = np.clip(np.array(accum_val), 0, 255).astype(np.uint8)
                # Else: No valid contribution (e.g., x_src, y_src too close to border for 4x4), return None

        except IndexError: # Catch potential indexing errors near borders
            # print(f"Index Error at src: ({x_src:.2f}, {y_src:.2f})") # Debug
            pass # Return None
            
        return pixel_value
