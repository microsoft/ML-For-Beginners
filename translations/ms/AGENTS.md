<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "93fdaa0fd38836e50c4793e2f2f25e8b",
  "translation_date": "2025-10-03T11:13:46+00:00",
  "source_file": "AGENTS.md",
  "language_code": "ms"
}
-->
# AGENTS.md

## Gambaran Projek

Ini adalah **Pembelajaran Mesin untuk Pemula**, kurikulum komprehensif selama 12 minggu dengan 26 pelajaran yang merangkumi konsep pembelajaran mesin klasik menggunakan Python (terutamanya dengan Scikit-learn) dan R. Repositori ini direka sebagai sumber pembelajaran kendiri dengan projek praktikal, kuiz, dan tugasan. Setiap pelajaran meneroka konsep ML melalui data dunia nyata dari pelbagai budaya dan wilayah di seluruh dunia.

Komponen utama:
- **Kandungan Pendidikan**: 26 pelajaran yang merangkumi pengenalan kepada ML, regresi, klasifikasi, pengelompokan, NLP, siri masa, dan pembelajaran pengukuhan
- **Aplikasi Kuiz**: Aplikasi kuiz berasaskan Vue.js dengan penilaian sebelum dan selepas pelajaran
- **Sokongan Pelbagai Bahasa**: Terjemahan automatik ke lebih 40 bahasa melalui GitHub Actions
- **Sokongan Dua Bahasa**: Pelajaran tersedia dalam Python (notebook Jupyter) dan R (fail R Markdown)
- **Pembelajaran Berasaskan Projek**: Setiap topik termasuk projek praktikal dan tugasan

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

Setiap folder pelajaran biasanya mengandungi:
- `README.md` - Kandungan utama pelajaran
- `notebook.ipynb` - Notebook Jupyter Python
- `solution/` - Kod penyelesaian (versi Python dan R)
- `assignment.md` - Latihan praktikal
- `images/` - Sumber visual

## Perintah Persediaan

### Untuk Pelajaran Python

Kebanyakan pelajaran menggunakan notebook Jupyter. Pasang keperluan yang diperlukan:

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

Pelajaran R berada dalam folder `solution/R/` sebagai fail `.rmd` atau `.ipynb`:

```bash
# Install R and required packages
# In R console:
install.packages(c("tidyverse", "tidymodels", "caret"))
```

### Untuk Aplikasi Kuiz

Aplikasi kuiz adalah aplikasi Vue.js yang terletak dalam direktori `quiz-app/`:

```bash
cd quiz-app
npm install
```

### Untuk Laman Dokumentasi

Untuk menjalankan dokumentasi secara tempatan:

```bash
# Install Docsify
npm install -g docsify-cli

# Serve from repository root
docsify serve

# Access at http://localhost:3000
```

## Aliran Kerja Pembangunan

### Bekerja dengan Notebook Pelajaran

1. Navigasi ke direktori pelajaran (contohnya, `2-Regression/1-Tools/`)
2. Buka notebook Jupyter:
   ```bash
   jupyter notebook notebook.ipynb
   ```
3. Kerjakan kandungan pelajaran dan latihan
4. Semak penyelesaian dalam folder `solution/` jika diperlukan

### Pembangunan Python

- Pelajaran menggunakan perpustakaan sains data Python standard
- Notebook Jupyter untuk pembelajaran interaktif
- Kod penyelesaian tersedia dalam folder `solution/` setiap pelajaran

### Pembangunan R

- Pelajaran R dalam format `.rmd` (R Markdown)
- Penyelesaian terletak dalam subdirektori `solution/R/`
- Gunakan RStudio atau Jupyter dengan kernel R untuk menjalankan notebook R

### Pembangunan Aplikasi Kuiz

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

## Arahan Pengujian

### Pengujian Aplikasi Kuiz

```bash
cd quiz-app

# Lint code
npm run lint

# Build to verify no errors
npm run build
```

**Nota**: Ini adalah repositori kurikulum pendidikan. Tiada ujian automatik untuk kandungan pelajaran. Pengesahan dilakukan melalui:
- Menyelesaikan latihan pelajaran
- Menjalankan sel notebook dengan berjaya
- Memeriksa output terhadap hasil yang dijangkakan dalam penyelesaian

## Garis Panduan Gaya Kod

### Kod Python
- Ikuti garis panduan gaya PEP 8
- Gunakan nama pemboleh ubah yang jelas dan deskriptif
- Sertakan komen untuk operasi yang kompleks
- Notebook Jupyter harus mempunyai sel markdown yang menerangkan konsep

### JavaScript/Vue.js (Aplikasi Kuiz)
- Mengikuti panduan gaya Vue.js
- Konfigurasi ESLint dalam `quiz-app/package.json`
- Jalankan `npm run lint` untuk memeriksa dan membetulkan isu secara automatik

### Dokumentasi
- Fail markdown harus jelas dan berstruktur dengan baik
- Sertakan contoh kod dalam blok kod berpagar
- Gunakan pautan relatif untuk rujukan dalaman
- Ikuti konvensyen format yang sedia ada

## Binaan dan Penerapan

### Penerapan Aplikasi Kuiz

Aplikasi kuiz boleh diterapkan ke Azure Static Web Apps:

1. **Prasyarat**:
   - Akaun Azure
   - Repositori GitHub (sudah difork)

2. **Terapkan ke Azure**:
   - Buat sumber Azure Static Web App
   - Sambungkan ke repositori GitHub
   - Tetapkan lokasi aplikasi: `/quiz-app`
   - Tetapkan lokasi output: `dist`
   - Azure secara automatik mencipta aliran kerja GitHub Actions

3. **Aliran Kerja GitHub Actions**:
   - Fail aliran kerja dicipta di `.github/workflows/azure-static-web-apps-*.yml`
   - Secara automatik membina dan menerapkan pada push ke cabang utama

### Dokumentasi PDF

Hasilkan PDF daripada dokumentasi:

```bash
npm install
npm run convert
```

## Aliran Kerja Terjemahan

**Penting**: Terjemahan dilakukan secara automatik melalui GitHub Actions menggunakan Co-op Translator.

- Terjemahan dihasilkan secara automatik apabila perubahan ditolak ke cabang `main`
- **JANGAN terjemahkan kandungan secara manual** - sistem akan menguruskan ini
- Aliran kerja ditakrifkan dalam `.github/workflows/co-op-translator.yml`
- Menggunakan perkhidmatan Azure AI/OpenAI untuk terjemahan
- Menyokong lebih 40 bahasa

## Garis Panduan Penyumbangan

### Untuk Penyumbang Kandungan

1. **Fork repositori** dan buat cabang ciri
2. **Buat perubahan pada kandungan pelajaran** jika menambah/mengemas kini pelajaran
3. **Jangan ubah fail terjemahan** - ia dihasilkan secara automatik
4. **Uji kod anda** - pastikan semua sel notebook berjalan dengan berjaya
5. **Sahkan pautan dan imej** berfungsi dengan betul
6. **Hantar permintaan tarik** dengan penerangan yang jelas

### Garis Panduan Permintaan Tarik

- **Format tajuk**: `[Bahagian] Penerangan ringkas tentang perubahan`
  - Contoh: `[Regression] Betulkan kesalahan ejaan dalam pelajaran 5`
  - Contoh: `[Quiz-App] Kemas kini kebergantungan`
- **Sebelum menghantar**:
  - Pastikan semua sel notebook berjalan tanpa ralat
  - Jalankan `npm run lint` jika mengubah aplikasi kuiz
  - Sahkan format markdown
  - Uji sebarang contoh kod baharu
- **PR mesti termasuk**:
  - Penerangan tentang perubahan
  - Sebab perubahan
  - Tangkapan skrin jika perubahan UI
- **Kod Etika**: Ikuti [Kod Etika Sumber Terbuka Microsoft](CODE_OF_CONDUCT.md)
- **CLA**: Anda perlu menandatangani Perjanjian Lesen Penyumbang

## Struktur Pelajaran

Setiap pelajaran mengikuti pola yang konsisten:

1. **Kuiz pra-kuliah** - Uji pengetahuan asas
2. **Kandungan pelajaran** - Arahan dan penjelasan bertulis
3. **Demonstrasi kod** - Contoh praktikal dalam notebook
4. **Pemeriksaan pengetahuan** - Sahkan pemahaman sepanjang pelajaran
5. **Cabaran** - Terapkan konsep secara bebas
6. **Tugasan** - Latihan lanjutan
7. **Kuiz pasca-kuliah** - Menilai hasil pembelajaran

## Rujukan Perintah Biasa

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

## Sumber Tambahan

- **Koleksi Microsoft Learn**: [Modul ML untuk Pemula](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)
- **Aplikasi Kuiz**: [Kuiz dalam talian](https://ff-quizzes.netlify.app/en/ml/)
- **Papan Perbincangan**: [Perbincangan GitHub](https://github.com/microsoft/ML-For-Beginners/discussions)
- **Panduan Video**: [Senarai Main YouTube](https://aka.ms/ml-beginners-videos)

## Teknologi Utama

- **Python**: Bahasa utama untuk pelajaran ML (Scikit-learn, Pandas, NumPy, Matplotlib)
- **R**: Pelaksanaan alternatif menggunakan tidyverse, tidymodels, caret
- **Jupyter**: Notebook interaktif untuk pelajaran Python
- **R Markdown**: Dokumen untuk pelajaran R
- **Vue.js 3**: Kerangka aplikasi kuiz
- **Flask**: Kerangka aplikasi web untuk penerapan model ML
- **Docsify**: Penjana laman dokumentasi
- **GitHub Actions**: CI/CD dan terjemahan automatik

## Pertimbangan Keselamatan

- **Tiada rahsia dalam kod**: Jangan sekali-kali komit kunci API atau kelayakan
- **Kebergantungan**: Pastikan pakej npm dan pip dikemas kini
- **Input pengguna**: Contoh aplikasi web Flask termasuk pengesahan input asas
- **Data sensitif**: Set data contoh adalah awam dan tidak sensitif

## Penyelesaian Masalah

### Notebook Jupyter

- **Isu kernel**: Mulakan semula kernel jika sel tergantung: Kernel → Restart
- **Ralat import**: Pastikan semua pakej yang diperlukan dipasang dengan pip
- **Isu laluan**: Jalankan notebook dari direktori yang mengandunginya

### Aplikasi Kuiz

- **npm install gagal**: Kosongkan cache npm: `npm cache clean --force`
- **Konflik port**: Tukar port dengan: `npm run serve -- --port 8081`
- **Ralat binaan**: Padamkan `node_modules` dan pasang semula: `rm -rf node_modules && npm install`

### Pelajaran R

- **Pakej tidak ditemui**: Pasang dengan: `install.packages("package-name")`
- **Rendering RMarkdown**: Pastikan pakej rmarkdown dipasang
- **Isu kernel**: Mungkin perlu memasang IRkernel untuk Jupyter

## Nota Khusus Projek

- Ini adalah **kurikulum pembelajaran**, bukan kod pengeluaran
- Fokus adalah pada **memahami konsep ML** melalui latihan praktikal
- Contoh kod mengutamakan **kejelasan berbanding pengoptimuman**
- Kebanyakan pelajaran adalah **berdiri sendiri** dan boleh diselesaikan secara bebas
- **Penyelesaian disediakan** tetapi pelajar harus mencuba latihan terlebih dahulu
- Repositori menggunakan **Docsify** untuk dokumentasi web tanpa langkah binaan
- **Sketchnotes** menyediakan ringkasan visual konsep
- **Sokongan pelbagai bahasa** menjadikan kandungan boleh diakses secara global

---

**Penafian**:  
Dokumen ini telah diterjemahkan menggunakan perkhidmatan terjemahan AI [Co-op Translator](https://github.com/Azure/co-op-translator). Walaupun kami berusaha untuk memastikan ketepatan, sila ambil perhatian bahawa terjemahan automatik mungkin mengandungi kesilapan atau ketidaktepatan. Dokumen asal dalam bahasa asalnya harus dianggap sebagai sumber yang berwibawa. Untuk maklumat yang kritikal, terjemahan manusia profesional adalah disyorkan. Kami tidak bertanggungjawab atas sebarang salah faham atau salah tafsir yang timbul daripada penggunaan terjemahan ini.