<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "93fdaa0fd38836e50c4793e2f2f25e8b",
  "translation_date": "2025-10-03T11:13:17+00:00",
  "source_file": "AGENTS.md",
  "language_code": "id"
}
-->
# AGENTS.md

## Gambaran Proyek

Ini adalah **Machine Learning untuk Pemula**, kurikulum komprehensif selama 12 minggu dengan 26 pelajaran yang mencakup konsep pembelajaran mesin klasik menggunakan Python (terutama dengan Scikit-learn) dan R. Repositori ini dirancang sebagai sumber belajar mandiri dengan proyek langsung, kuis, dan tugas. Setiap pelajaran mengeksplorasi konsep ML melalui data dunia nyata dari berbagai budaya dan wilayah di seluruh dunia.

Komponen utama:
- **Konten Edukasi**: 26 pelajaran yang mencakup pengenalan ML, regresi, klasifikasi, clustering, NLP, time series, dan pembelajaran penguatan
- **Aplikasi Kuis**: Aplikasi kuis berbasis Vue.js dengan penilaian sebelum dan sesudah pelajaran
- **Dukungan Multi-bahasa**: Terjemahan otomatis ke lebih dari 40 bahasa melalui GitHub Actions
- **Dukungan Bahasa Ganda**: Pelajaran tersedia dalam Python (notebook Jupyter) dan R (file R Markdown)
- **Pembelajaran Berbasis Proyek**: Setiap topik mencakup proyek praktis dan tugas

## Struktur Repositori

```
ML-For-Beginners/
├── 1-Introduction/         # ML basics, history, fairness, techniques
├── 2-Regression/          # Regression models with Python/R
├── 3-Web-App/            # Flask web app for ML model deployment
├── 4-Classification/      # Classification algorithms
├── 5-Clustering/         # Clustering techniques
├── 6-NLP/               # Natural Language Processing
├── 7-TimeSeries/        # Time series forecasting
├── 8-Reinforcement/     # Reinforcement learning
├── 9-Real-World/        # Real-world ML applications
├── quiz-app/           # Vue.js quiz application
├── translations/       # Auto-generated translations
└── sketchnotes/       # Visual learning aids
```

Setiap folder pelajaran biasanya berisi:
- `README.md` - Konten utama pelajaran
- `notebook.ipynb` - Notebook Jupyter Python
- `solution/` - Kode solusi (versi Python dan R)
- `assignment.md` - Latihan praktik
- `images/` - Sumber daya visual

## Perintah Setup

### Untuk Pelajaran Python

Sebagian besar pelajaran menggunakan notebook Jupyter. Instal dependensi yang diperlukan:

```bash
# Install Python 3.8+ if not already installed
python --version

# Install Jupyter
pip install jupyter

# Install common ML libraries
pip install scikit-learn pandas numpy matplotlib seaborn

# For specific lessons, check lesson-specific requirements
# Example: Web App lesson
pip install flask
```

### Untuk Pelajaran R

Pelajaran R berada di folder `solution/R/` sebagai file `.rmd` atau `.ipynb`:

```bash
# Install R and required packages
# In R console:
install.packages(c("tidyverse", "tidymodels", "caret"))
```

### Untuk Aplikasi Kuis

Aplikasi kuis adalah aplikasi Vue.js yang terletak di direktori `quiz-app/`:

```bash
cd quiz-app
npm install
```

### Untuk Situs Dokumentasi

Untuk menjalankan dokumentasi secara lokal:

```bash
# Install Docsify
npm install -g docsify-cli

# Serve from repository root
docsify serve

# Access at http://localhost:3000
```

## Alur Kerja Pengembangan

### Bekerja dengan Notebook Pelajaran

1. Masuk ke direktori pelajaran (misalnya, `2-Regression/1-Tools/`)
2. Buka notebook Jupyter:
   ```bash
   jupyter notebook notebook.ipynb
   ```
3. Kerjakan konten pelajaran dan latihan
4. Periksa solusi di folder `solution/` jika diperlukan

### Pengembangan Python

- Pelajaran menggunakan pustaka data science Python standar
- Notebook Jupyter untuk pembelajaran interaktif
- Kode solusi tersedia di folder `solution/` setiap pelajaran

### Pengembangan R

- Pelajaran R dalam format `.rmd` (R Markdown)
- Solusi terletak di subdirektori `solution/R/`
- Gunakan RStudio atau Jupyter dengan kernel R untuk menjalankan notebook R

### Pengembangan Aplikasi Kuis

```bash
cd quiz-app

# Start development server
npm run serve
# Access at http://localhost:8080

# Build for production
npm run build

# Lint and fix files
npm run lint
```

## Instruksi Pengujian

### Pengujian Aplikasi Kuis

```bash
cd quiz-app

# Lint code
npm run lint

# Build to verify no errors
npm run build
```

**Catatan**: Ini terutama repositori kurikulum edukasi. Tidak ada pengujian otomatis untuk konten pelajaran. Validasi dilakukan melalui:
- Menyelesaikan latihan pelajaran
- Menjalankan sel notebook dengan sukses
- Memeriksa output terhadap hasil yang diharapkan di solusi

## Panduan Gaya Kode

### Kode Python
- Ikuti panduan gaya PEP 8
- Gunakan nama variabel yang jelas dan deskriptif
- Sertakan komentar untuk operasi yang kompleks
- Notebook Jupyter harus memiliki sel markdown yang menjelaskan konsep

### JavaScript/Vue.js (Aplikasi Kuis)
- Mengikuti panduan gaya Vue.js
- Konfigurasi ESLint di `quiz-app/package.json`
- Jalankan `npm run lint` untuk memeriksa dan memperbaiki masalah secara otomatis

### Dokumentasi
- File markdown harus jelas dan terstruktur dengan baik
- Sertakan contoh kode dalam blok kode berpagar
- Gunakan tautan relatif untuk referensi internal
- Ikuti konvensi format yang ada

## Build dan Deployment

### Deployment Aplikasi Kuis

Aplikasi kuis dapat dideploy ke Azure Static Web Apps:

1. **Prasyarat**:
   - Akun Azure
   - Repositori GitHub (sudah di-fork)

2. **Deploy ke Azure**:
   - Buat sumber daya Azure Static Web App
   - Hubungkan ke repositori GitHub
   - Tetapkan lokasi aplikasi: `/quiz-app`
   - Tetapkan lokasi output: `dist`
   - Azure secara otomatis membuat workflow GitHub Actions

3. **Workflow GitHub Actions**:
   - File workflow dibuat di `.github/workflows/azure-static-web-apps-*.yml`
   - Secara otomatis membangun dan mendeply saat ada push ke branch utama

### Dokumentasi PDF

Hasilkan PDF dari dokumentasi:

```bash
npm install
npm run convert
```

## Alur Kerja Terjemahan

**Penting**: Terjemahan dilakukan secara otomatis melalui GitHub Actions menggunakan Co-op Translator.

- Terjemahan dibuat secara otomatis saat ada perubahan yang di-push ke branch `main`
- **JANGAN menerjemahkan konten secara manual** - sistem menangani ini
- Workflow didefinisikan di `.github/workflows/co-op-translator.yml`
- Menggunakan layanan Azure AI/OpenAI untuk terjemahan
- Mendukung lebih dari 40 bahasa

## Panduan Kontribusi

### Untuk Kontributor Konten

1. **Fork repositori** dan buat branch fitur
2. **Lakukan perubahan pada konten pelajaran** jika menambahkan/memperbarui pelajaran
3. **Jangan mengubah file yang diterjemahkan** - file tersebut dibuat secara otomatis
4. **Uji kode Anda** - pastikan semua sel notebook berjalan dengan sukses
5. **Verifikasi tautan dan gambar** berfungsi dengan benar
6. **Kirim pull request** dengan deskripsi yang jelas

### Panduan Pull Request

- **Format judul**: `[Bagian] Deskripsi singkat perubahan`
  - Contoh: `[Regression] Perbaiki typo di pelajaran 5`
  - Contoh: `[Quiz-App] Perbarui dependensi`
- **Sebelum mengirimkan**:
  - Pastikan semua sel notebook berjalan tanpa error
  - Jalankan `npm run lint` jika memodifikasi quiz-app
  - Verifikasi format markdown
  - Uji contoh kode baru
- **PR harus mencakup**:
  - Deskripsi perubahan
  - Alasan perubahan
  - Screenshot jika ada perubahan UI
- **Kode Etik**: Ikuti [Kode Etik Sumber Terbuka Microsoft](CODE_OF_CONDUCT.md)
- **CLA**: Anda perlu menandatangani Perjanjian Lisensi Kontributor

## Struktur Pelajaran

Setiap pelajaran mengikuti pola yang konsisten:

1. **Kuis pra-pelajaran** - Menguji pengetahuan awal
2. **Konten pelajaran** - Instruksi dan penjelasan tertulis
3. **Demonstrasi kode** - Contoh langsung dalam notebook
4. **Pemeriksaan pengetahuan** - Memverifikasi pemahaman sepanjang pelajaran
5. **Tantangan** - Menerapkan konsep secara mandiri
6. **Tugas** - Latihan lanjutan
7. **Kuis pasca-pelajaran** - Menilai hasil pembelajaran

## Referensi Perintah Umum

```bash
# Python/Jupyter
jupyter notebook                    # Start Jupyter server
jupyter notebook notebook.ipynb     # Open specific notebook
pip install -r requirements.txt     # Install dependencies (where available)

# Quiz App
cd quiz-app
npm install                        # Install dependencies
npm run serve                      # Development server
npm run build                      # Production build
npm run lint                       # Lint and fix

# Documentation
docsify serve                      # Serve documentation locally
npm run convert                    # Generate PDF

# Git workflow
git checkout -b feature/my-change  # Create feature branch
git add .                         # Stage changes
git commit -m "Description"       # Commit changes
git push origin feature/my-change # Push to remote
```

## Sumber Daya Tambahan

- **Koleksi Microsoft Learn**: [Modul ML untuk Pemula](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)
- **Aplikasi Kuis**: [Kuis online](https://ff-quizzes.netlify.app/en/ml/)
- **Forum Diskusi**: [Diskusi GitHub](https://github.com/microsoft/ML-For-Beginners/discussions)
- **Panduan Video**: [Playlist YouTube](https://aka.ms/ml-beginners-videos)

## Teknologi Utama

- **Python**: Bahasa utama untuk pelajaran ML (Scikit-learn, Pandas, NumPy, Matplotlib)
- **R**: Implementasi alternatif menggunakan tidyverse, tidymodels, caret
- **Jupyter**: Notebook interaktif untuk pelajaran Python
- **R Markdown**: Dokumen untuk pelajaran R
- **Vue.js 3**: Kerangka kerja aplikasi kuis
- **Flask**: Kerangka kerja aplikasi web untuk deployment model ML
- **Docsify**: Generator situs dokumentasi
- **GitHub Actions**: CI/CD dan terjemahan otomatis

## Pertimbangan Keamanan

- **Tidak ada rahasia dalam kode**: Jangan pernah menyertakan API key atau kredensial
- **Dependensi**: Jaga agar paket npm dan pip tetap diperbarui
- **Input pengguna**: Contoh aplikasi web Flask mencakup validasi input dasar
- **Data sensitif**: Dataset contoh bersifat publik dan tidak sensitif

## Pemecahan Masalah

### Notebook Jupyter

- **Masalah kernel**: Restart kernel jika sel macet: Kernel → Restart
- **Error impor**: Pastikan semua paket yang diperlukan diinstal dengan pip
- **Masalah path**: Jalankan notebook dari direktori tempatnya berada

### Aplikasi Kuis

- **npm install gagal**: Bersihkan cache npm: `npm cache clean --force`
- **Konflik port**: Ubah port dengan: `npm run serve -- --port 8081`
- **Error build**: Hapus `node_modules` dan instal ulang: `rm -rf node_modules && npm install`

### Pelajaran R

- **Paket tidak ditemukan**: Instal dengan: `install.packages("nama-paket")`
- **Rendering RMarkdown**: Pastikan paket rmarkdown diinstal
- **Masalah kernel**: Mungkin perlu menginstal IRkernel untuk Jupyter

## Catatan Khusus Proyek

- Ini terutama merupakan **kurikulum pembelajaran**, bukan kode produksi
- Fokus pada **pemahaman konsep ML** melalui praktik langsung
- Contoh kode memprioritaskan **kejelasan daripada optimasi**
- Sebagian besar pelajaran **mandiri** dan dapat diselesaikan secara independen
- **Solusi disediakan**, tetapi pembelajar harus mencoba latihan terlebih dahulu
- Repositori menggunakan **Docsify** untuk dokumentasi web tanpa langkah build
- **Sketchnotes** menyediakan ringkasan visual konsep
- **Dukungan multi-bahasa** membuat konten dapat diakses secara global

---

**Penafian**:  
Dokumen ini telah diterjemahkan menggunakan layanan penerjemahan AI [Co-op Translator](https://github.com/Azure/co-op-translator). Meskipun kami berupaya untuk memberikan hasil yang akurat, harap diperhatikan bahwa terjemahan otomatis mungkin mengandung kesalahan atau ketidakakuratan. Dokumen asli dalam bahasa aslinya harus dianggap sebagai sumber yang otoritatif. Untuk informasi yang bersifat kritis, disarankan menggunakan jasa penerjemahan manusia profesional. Kami tidak bertanggung jawab atas kesalahpahaman atau penafsiran yang keliru yang timbul dari penggunaan terjemahan ini.