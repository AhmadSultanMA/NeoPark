## 🚗 Neopark - Sistem Manajemen Parkir Pintar Berbasis AI

Neopark adalah prototipe sistem manajemen parkir pintar yang memanfaatkan Kecerdasan Buatan (AI) untuk deteksi objek dan Internet of Things (IoT) untuk pemantauan secara real-time. Proyek ini dirancang untuk memberikan informasi ketersediaan slot parkir secara akurat dan efisien melalui sebuah dashboard web interaktif.

Proyek ini telah berhasil didemonstrasikan dalam sebuah pameran teknologi dan memenangkan penghargaan sebagai **Booth Paling Banyak Pengunjungnya**, menunjukkan tingginya minat dan relevansi solusi parkir pintar di era modern.

## 🏛️ Arsitektur dan Penerapan Sistem Terdistribusi

Meskipun terlihat sebagai satu kesatuan, proyek Neopark secara fundamental diimplementasikan sebagai sebuah sistem terdistribusi. Ini berarti sistem terdiri dari beberapa komponen independen yang saling terhubung dan berkomunikasi melalui jaringan untuk mencapai tujuan bersama.

### Pemecahan Arsitektur Terdistribusi

#### Nodes Sensor/Data (ESP32-CAM)

-   **Tugas**: Bertanggung jawab penuh untuk satu hal, yaitu akuisisi data visual (gambar).
-   **Komunikasi**: Mengirimkan data gambar secara terus-menerus ke server pusat melalui jaringan lokal (WiFi) menggunakan protokol HTTP.
-   **Peran**: Merupakan "edge devices" dalam sistem ini.

#### Node Pemrosesan Pusat (Server Flask & YOLOv8)

-   **Tugas**:
    -   Menerima dan mengelola aliran data dari berbagai node sensor (ESP32-CAM).
    -   Menjalankan tugas komputasi berat, yaitu inferensi model AI (YOLOv8) untuk mendeteksi objek.
    -   Mengagregasi hasil deteksi dan mengelola status sistem (ketersediaan slot, status koneksi).
    -   Menyediakan API untuk diakses oleh komponen lain (klien dan monitoring).

#### Node Klien (Dashboard Website)

-   **Tugas**: Menyajikan data yang telah diproses kepada pengguna akhir dalam bentuk visual yang mudah dipahami.
-   **Komunikasi**: Berkomunikasi dengan server melalui API HTTP untuk mendapatkan data dan menampilkan informasi.

#### Nodes Monitoring (Prometheus & Grafana)

-   **Prometheus**: Menarik data metrik dari endpoint `/metrics` yang disediakan oleh server aplikasi.
-   **Grafana**: Menarik data dari Prometheus untuk divisualisasikan dan memberikan wawasan tentang kesehatan dan kinerja sistem.

Secara keseluruhan, pemisahan antara akuisisi data, pemrosesan, penyajian, dan monitoring menjadi komponen-komponen yang saling berkomunikasi melalui jaringan adalah inti dari penerapan konsep sistem terdistribusi pada proyek Neopark.

## ✨ Fitur Utama

-   **Deteksi Okupansi Parkir Real-time**: Menggunakan model AI YOLOv8 untuk mendeteksi mobil dan menghitung slot yang terisi/tersedia.
-   **Pemantauan Multi-Area**: Mampu memantau lebih dari satu area parkir (A1, A2) secara simultan.
-   **Dashboard Web Interaktif**: Menampilkan data secara visual, termasuk:
    -   Jumlah slot tersedia dan terisi per area dan total.
    -   Live video feed dengan bounding box hasil deteksi.
    -   Status koneksi kamera.
-   **Monitoring Sistem**: Dashboard Grafana untuk memantau metrik performa server, okupansi parkir, dan confidence score model.
-   **Arsitektur Berbasis Kontainer**: Seluruh layanan diatur menggunakan Docker dan Docker Compose untuk kemudahan deployment dan portabilitas.

## 🛠️ Tumpukan Teknologi (Technology Stack)

### AI & Machine Learning

-   **YOLOv8**: Model deteksi objek.
-   **PyTorch**: Framework dasar untuk YOLOv8.
-   **Ultralytics**: Framework untuk menggunakan YOLOv8.

### Perangkat Keras (IoT)

-   **ESP32-CAM**: Untuk pengambilan dan streaming gambar.

### Backend

-   **Python**: Bahasa pemrograman utama.
-   **Flask**: Web framework untuk membangun API server.

### Frontend

-   **HTML, CSS, JavaScript**: Untuk membangun dashboard web.

### DevOps & Infrastruktur

-   **Docker & Docker Compose**: Untuk kontainerisasi dan orkestrasi layanan.
-   **Nginx**: Sebagai reverse proxy (jika digunakan).
-   **GitHub Actions**: Untuk Continuous Integration (CI).
-   **Vercel**: Untuk Continuous Deployment (CD) frontend.

### Monitoring

-   **Prometheus**: Untuk pengumpulan dan penyimpanan data metrik.
-   **Grafana**: Untuk visualisasi dan pembuatan dashboard.

## 🚀 Pengaturan dan Instalasi

Proyek ini dirancang untuk dijalankan dengan mudah menggunakan Docker dan Docker Compose.

### Prasyarat

-   **Docker**: Instal Docker
-   **Docker Compose**: Biasanya sudah termasuk dalam instalasi Docker Desktop.
-   **Git**: Untuk mengkloning repositori.

### Langkah-langkah Menjalankan

#### 1. Kloning Repositori

```bash
git clone https://github.com/AhmadSultanMA/NeoPark.git
cd NeoPark
```

````

#### 2. Konfigurasi Perangkat IoT (ESP32-CAM)

-   Flash firmware yang ada di folder `Arduino/ESP32CAM` ke perangkat ESP32-CAM Anda.
-   Pastikan perangkat terhubung ke jaringan WiFi yang sama dengan laptop server.
-   Ubah kode firmware untuk mengirimkan stream gambar ke alamat IP laptop server Anda, port 5000 (misalnya, `http://192.168.1.100:5000/a1/upload`).

#### 3. Jalankan Semua Layanan dengan Docker Compose

Pastikan Anda berada di direktori root proyek (yang berisi file `docker-compose.yml`).

```bash
docker-compose up -d --build
```

Perintah ini akan:

-   Membangun Docker image untuk `neopark-server`.
-   Menjalankan semua layanan yang didefinisikan di `docker-compose.yml` (neopark-server, nginx, prometheus, grafana) di latar belakang.

#### 4. Akses Layanan

-   **Dashboard Neopark**: Buka `http://localhost` atau `http://<IP-Laptop-Anda>`.
-   **Grafana**: Buka `http://localhost:3000` (login default: `admin/admin`).
-   **Prometheus**: Buka `http://localhost:9090`.

#### 5. Menghentikan Layanan

Untuk menghentikan semua kontainer yang berjalan:

```bash
docker-compose down
```

## ⚙️ CI/CD & Monitoring

### Continuous Integration

Setiap push atau pull request ke branch main akan memicu workflow **GitHub Actions** yang melakukan linting (flake8) dan pengujian (pytest) pada kode backend.

### Continuous Deployment

Setiap push dengan pesan commit yang mengandung **"website commit"** akan secara otomatis men-deploy frontend ke Vercel.

### Monitoring

Metrik dari server aplikasi diekspos pada endpoint `/metrics` dan di-scrape oleh Prometheus setiap 15 detik. Data ini dapat divisualisasikan di Grafana untuk memantau kesehatan sistem.

## 📂 Struktur Direktori

```bash
NeoPark/
├── .github/workflows/          # Konfigurasi CI GitHub Actions
├── Arduino/                    # Kode firmware untuk perangkat IoT
│   ├── ESP32/
│   └── ESP32CAM/
├── FineTune-YOLOV8/            # Skrip dan data untuk fine-tuning model
├── Server/                     # Kode backend Flask
│   ├── neopark_server.py
│   └── fine-best.pt            # File model AI
├── tests/                      # Tes otomatis untuk server
│   ├── conftest.py
│   ├── test_server_api.py
│   └── test_server_helpers.py
├── Website/                    # File frontend
│   └── neopark-dashboard.html
├── .dockerignore
├── .flake8                     # Konfigurasi linter
├── docker-compose.yml          # Orkestrasi layanan Docker
├── Dockerfile                  # Definisi image Docker untuk server
├── nginx.conf                  # Konfigurasi Nginx
├── requirements.txt            # Dependensi Python untuk server
└── README.md                   # File ini
```

## 🔮 Pengembangan Selanjutnya

-   Validasi sistem pada skala nyata dengan kondisi lingkungan yang beragam.
-   Pengembangan fitur reservasi dan pembayaran parkir.
-   Implementasi model prediksi untuk memperkirakan ketersediaan parkir di masa depan.
-   Optimasi model AI untuk dijalankan pada perangkat edge dengan sumber daya terbatas.
````
