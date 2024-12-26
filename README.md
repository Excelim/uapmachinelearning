# 🌱 Aplikasi Prediksi Penyakit Daun Palam

### 👤 Nama: Excel Bima Evansyah
### 🌐 NIM: 202110370311006

---

## 🔍 Table of Contents

1. [Deskripsi Proyek](#deskripsi-proyek)
2. [Dataset](#dataset)
3. [Repository](#repository)
4. [Langkah Instalasi](#langkah-instalasi)
5. [Deskripsi Model](#deskripsi-model)
6. [Kontak](#kontak)

---

## 🔎 Deskripsi Proyek <a id="deskripsi-proyek"></a>

Repositori ini berisi proyek pembelajaran mesin untuk mengklasifikasikan penyakit pada daun pohon palem. Dataset ini terdiri dari gambar daun palem yang menunjukkan kondisi yang berbeda, termasuk daun yang sehat dan daun yang terinfeksi berbagai hama atau penyakit.

---
## 📊 Dataset <a id="dataset"></a>

Dataset yang digunakan berasal dari Kaggle: [Production Dataset](https://www.kaggle.com/datasets/warcoder/palm-leaves-dataset).

## 🔧 Repository <a id="repository"></a>

Repository proyek dapat ditemukan di GitHub:[excelim/uapmachinelearning](https://github.com/Excelim/uapmachinelearning)

## 📚 Langkah Instalasi <a id="langkah-instalasi"></a>

Ikuti langkah-langkah berikut untuk menjalankan proyek ini secara lokal:

1. Clone repository ini:

   ```bash
   git clone https://github.com/Excelim/uapmachinelearning.git
   ```

2. Pindah ke direktori proyek:

   ```bash
   cd UAP_Machine_Learning
   ```

3. Buat virtual environment dan aktifkan:

   ```bash
   python -m venv venv
   source venv/bin/activate  # Untuk Linux/Mac
   venv\Scripts\activate   # Untuk Windows
   ```

4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
5. Unduh Model:

   [Model .pkl](https://drive.google.com/drive/folders/1Ps75rw-16Xkdd7FhtUWbup-2w4IYejfr?usp=sharing)

6. Jalankan aplikasi Streamlit:

   ```bash
   streamlit run app.py
   ```

---

## 🔢 Deskripsi Model <a id="deskripsi-model"></a>

Proyek ini menggunakan dua model:
1. **VGG16 (Fine-Tuning)**: Model VGG16 yang sudah dilatih sebelumnya digunakan sebagai dasar dan disesuaikan untuk tugas klasifikasi penyakit daun palem.
2. **CNN Kustom**: Model CNN sederhana yang dibangun dari awal untuk mengklasifikasikan daun palem.

## 📢 Kontak <a id="kontak"></a>

Terima kasih telah menggunakan aplikasi ini! Jika Anda memiliki pertanyaan atau saran, jangan ragu untuk membuka isu di repository GitHub.


