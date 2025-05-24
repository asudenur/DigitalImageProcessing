# Dijital Görüntü İşleme

Bu proje, Python, PyQt5 ve OpenCV kullanılarak geliştirilmiş kapsamlı bir dijital görüntü işleme uygulamasıdır. Kullanıcı dostu grafiksel arayüzü aracılığıyla çeşitli görüntü işleme teknikleri ve algoritmaları uygular.

### Özellikler
1. **Temel İşlevsellik (Ödev 1)**
   - Görüntü yükleme ve kaydetme
   - Gri tonlamaya dönüştürme
   - Parlaklık ve kontrast ayarlama

2. **Temel Görüntü Operasyonları ve İnterpolasyon (Ödev 2)**
   - Görüntü büyütme ve küçültme
   - Yakınlaştırma/uzaklaştırma işlevselliği
   - Görüntü döndürme
   - Çoklu interpolasyon yöntemleri:
     - Bilinear
     - Bicubic
     - Ortalama

3. **Final Projesi Özellikleri**
   - S-Curve Kontrast Güçlendirme
     ![Ekran görüntüsü 2025-05-23 171706](https://github.com/user-attachments/assets/7ecad4d3-1e22-4aa5-b365-f5a6f9645306)
     ![Ekran görüntüsü 2025-05-23 171739](https://github.com/user-attachments/assets/72937f1f-da79-4733-ab22-3d14e143220c)
     ![Ekran görüntüsü 2025-05-23 171729](https://github.com/user-attachments/assets/666c4d5d-226e-402c-b67e-cfddfa0eb5ea)
     ![Ekran görüntüsü 2025-05-23 171719](https://github.com/user-attachments/assets/8cce0dd6-67b0-4180-93c6-0150a1d5ce89)

   - Hough Dönüşümü Uygulamaları
     - Yol Çizgi tespiti
     ![Ekran görüntüsü 2025-05-23 171618](https://github.com/user-attachments/assets/72cedf73-1755-470b-872c-077d920badcf)
     - Göz tespiti
     ![Ekran görüntüsü 2025-05-23 171605](https://github.com/user-attachments/assets/6fe0f221-0ffa-4d2b-a8fa-575fb84d7829)
   - Hareket Bulanıklığı Düzeltme (Deblurring)
     ![Ekran görüntüsü 2025-05-23 171633](https://github.com/user-attachments/assets/d41f3d8c-def2-4e84-836d-db875b9e17db)
   - Nesne Sayma ve Özellik Çıkarma
     - Nesne tespiti
     - Özellik analizi
     - Excel'e aktarma işlevselliği     
    ![Ekran görüntüsü 2025-05-23 171651](https://github.com/user-attachments/assets/963ecfac-7cd9-49d4-95de-db45b44f97a0)

### Gereksinimler
- Python 3.x
- PyQt5
- OpenCV (cv2)
- NumPy
- Pandas

### Kurulum
1. Depoyu klonlayın:
```bash
git clone [repository-url]
```

2. Sanal ortam oluşturun ve aktifleştirin:
```bash
python -m venv venv
source venv/bin/activate  # Windows'ta: venv\Scripts\activate
```

3. Gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```

### Kullanım
Ana uygulamayı çalıştırın:
```bash
python venv/main.py
```

### Proje Yapısı
- `main.py`: Ana uygulama penceresi ve temel işlevsellik
- `odev2.py`: Temel görüntü operasyonları ve interpolasyon uygulaması
- `finalodevi.py`: Gelişmiş görüntü işleme özelliklerinin uygulaması
