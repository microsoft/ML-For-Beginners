<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "134d8759f0e2ab886e9aa4f62362c201",
  "translation_date": "2025-10-03T12:50:48+00:00",
  "source_file": "TROUBLESHOOTING.md",
  "language_code": "id"
}
-->
# Panduan Pemecahan Masalah

Panduan ini membantu Anda menyelesaikan masalah umum saat bekerja dengan kurikulum Machine Learning untuk Pemula. Jika Anda tidak menemukan solusi di sini, silakan cek [Diskusi Discord](https://aka.ms/foundry/discord) atau [buka masalah baru](https://github.com/microsoft/ML-For-Beginners/issues).

## Daftar Isi

- [Masalah Instalasi](../..)
- [Masalah Jupyter Notebook](../..)
- [Masalah Paket Python](../..)
- [Masalah Lingkungan R](../..)
- [Masalah Aplikasi Kuis](../..)
- [Masalah Data dan Jalur File](../..)
- [Pesan Kesalahan Umum](../..)
- [Masalah Performa](../..)
- [Lingkungan dan Konfigurasi](../..)

---

## Masalah Instalasi

### Instalasi Python

**Masalah**: `python: command not found`

**Solusi**:
1. Instal Python 3.8 atau versi lebih tinggi dari [python.org](https://www.python.org/downloads/)
2. Verifikasi instalasi: `python --version` atau `python3 --version`
3. Di macOS/Linux, Anda mungkin perlu menggunakan `python3` alih-alih `python`

**Masalah**: Konflik karena beberapa versi Python

**Solusi**:
```bash
# Use virtual environments to isolate projects
python -m venv ml-env

# Activate virtual environment
# On Windows:
ml-env\Scripts\activate
# On macOS/Linux:
source ml-env/bin/activate
```

### Instalasi Jupyter

**Masalah**: `jupyter: command not found`

**Solusi**:
```bash
# Install Jupyter
pip install jupyter

# Or with pip3
pip3 install jupyter

# Verify installation
jupyter --version
```

**Masalah**: Jupyter tidak terbuka di browser

**Solusi**:
```bash
# Try specifying the browser
jupyter notebook --browser=chrome

# Or copy the URL with token from terminal and paste in browser manually
# Look for: http://localhost:8888/?token=...
```

### Instalasi R

**Masalah**: Paket R tidak dapat diinstal

**Solusi**:
```r
# Ensure you have the latest R version
# Install packages with dependencies
install.packages(c("tidyverse", "tidymodels", "caret"), dependencies = TRUE)

# If compilation fails, try installing binary versions
install.packages("package-name", type = "binary")
```

**Masalah**: IRkernel tidak tersedia di Jupyter

**Solusi**:
```r
# In R console
install.packages('IRkernel')
IRkernel::installspec(user = TRUE)
```

---

## Masalah Jupyter Notebook

### Masalah Kernel

**Masalah**: Kernel terus mati atau restart

**Solusi**:
1. Restart kernel: `Kernel → Restart`
2. Hapus output dan restart: `Kernel → Restart & Clear Output`
3. Periksa masalah memori (lihat [Masalah Performa](../..))
4. Coba jalankan sel satu per satu untuk mengidentifikasi kode yang bermasalah

**Masalah**: Kernel Python yang salah dipilih

**Solusi**:
1. Periksa kernel saat ini: `Kernel → Change Kernel`
2. Pilih versi Python yang benar
3. Jika kernel hilang, buat kernel baru:
```bash
python -m ipykernel install --user --name=ml-env
```

**Masalah**: Kernel tidak dapat dimulai

**Solusi**:
```bash
# Reinstall ipykernel
pip uninstall ipykernel
pip install ipykernel

# Register the kernel again
python -m ipykernel install --user
```

### Masalah Sel Notebook

**Masalah**: Sel berjalan tetapi tidak menunjukkan output

**Solusi**:
1. Periksa apakah sel masih berjalan (lihat indikator `[*]`)
2. Restart kernel dan jalankan semua sel: `Kernel → Restart & Run All`
3. Periksa konsol browser untuk kesalahan JavaScript (F12)

**Masalah**: Tidak dapat menjalankan sel - tidak ada respons saat mengklik "Run"

**Solusi**:
1. Periksa apakah server Jupyter masih berjalan di terminal
2. Refresh halaman browser
3. Tutup dan buka kembali notebook
4. Restart server Jupyter

---

## Masalah Paket Python

### Kesalahan Import

**Masalah**: `ModuleNotFoundError: No module named 'sklearn'`

**Solusi**:
```bash
pip install scikit-learn

# Common ML packages for this course
pip install scikit-learn pandas numpy matplotlib seaborn
```

**Masalah**: `ImportError: cannot import name 'X' from 'sklearn'`

**Solusi**:
```bash
# Update scikit-learn to latest version
pip install --upgrade scikit-learn

# Check version
python -c "import sklearn; print(sklearn.__version__)"
```

### Konflik Versi

**Masalah**: Kesalahan ketidakcocokan versi paket

**Solusi**:
```bash
# Create a new virtual environment
python -m venv fresh-env
source fresh-env/bin/activate  # or fresh-env\Scripts\activate on Windows

# Install packages fresh
pip install jupyter scikit-learn pandas numpy matplotlib seaborn

# If specific version needed
pip install scikit-learn==1.3.0
```

**Masalah**: `pip install` gagal dengan kesalahan izin

**Solusi**:
```bash
# Install for current user only
pip install --user package-name

# Or use virtual environment (recommended)
python -m venv venv
source venv/bin/activate
pip install package-name
```

### Masalah Pemrosesan Data

**Masalah**: `FileNotFoundError` saat memuat file CSV

**Solusi**:
```python
import os
# Check current working directory
print(os.getcwd())

# Use relative paths from notebook location
df = pd.read_csv('../../data/filename.csv')

# Or use absolute paths
df = pd.read_csv('/full/path/to/data/filename.csv')
```

---

## Masalah Lingkungan R

### Instalasi Paket

**Masalah**: Instalasi paket gagal dengan kesalahan kompilasi

**Solusi**:
```r
# Install binary version (Windows/macOS)
install.packages("package-name", type = "binary")

# Update R to latest version if packages require it
# Check R version
R.version.string

# Install system dependencies (Linux)
# For Ubuntu/Debian, in terminal:
# sudo apt-get install r-base-dev
```

**Masalah**: `tidyverse` tidak dapat diinstal

**Solusi**:
```r
# Install dependencies first
install.packages(c("rlang", "vctrs", "pillar"))

# Then install tidyverse
install.packages("tidyverse")

# Or install components individually
install.packages(c("dplyr", "ggplot2", "tidyr", "readr"))
```

### Masalah RMarkdown

**Masalah**: RMarkdown tidak dapat dirender

**Solusi**:
```r
# Install/update rmarkdown
install.packages("rmarkdown")

# Install pandoc if needed
install.packages("pandoc")

# For PDF output, install tinytex
install.packages("tinytex")
tinytex::install_tinytex()
```

---

## Masalah Aplikasi Kuis

### Build dan Instalasi

**Masalah**: `npm install` gagal

**Solusi**:
```bash
# Clear npm cache
npm cache clean --force

# Remove node_modules and package-lock.json
rm -rf node_modules package-lock.json

# Reinstall
npm install

# If still fails, try with legacy peer deps
npm install --legacy-peer-deps
```

**Masalah**: Port 8080 sudah digunakan

**Solusi**:
```bash
# Use different port
npm run serve -- --port 8081

# Or find and kill process using port 8080
# On Linux/macOS:
lsof -ti:8080 | xargs kill -9

# On Windows:
netstat -ano | findstr :8080
taskkill /PID <PID> /F
```

### Kesalahan Build

**Masalah**: `npm run build` gagal

**Solusi**:
```bash
# Check Node.js version (should be 14+)
node --version

# Update Node.js if needed
# Then clean install
rm -rf node_modules package-lock.json
npm install
npm run build
```

**Masalah**: Kesalahan linting mencegah build

**Solusi**:
```bash
# Fix auto-fixable issues
npm run lint -- --fix

# Or temporarily disable linting in build
# (not recommended for production)
```

---

## Masalah Data dan Jalur File

### Masalah Jalur

**Masalah**: File data tidak ditemukan saat menjalankan notebook

**Solusi**:
1. **Selalu jalankan notebook dari direktori tempat file berada**
   ```bash
   cd /path/to/lesson/folder
   jupyter notebook
   ```

2. **Periksa jalur relatif dalam kode**
   ```python
   # Correct path from notebook location
   df = pd.read_csv('../data/filename.csv')
   
   # Not from your terminal location
   ```

3. **Gunakan jalur absolut jika diperlukan**
   ```python
   import os
   base_path = os.path.dirname(os.path.abspath(__file__))
   data_path = os.path.join(base_path, 'data', 'filename.csv')
   ```

### File Data Hilang

**Masalah**: File dataset hilang

**Solusi**:
1. Periksa apakah data seharusnya ada di repositori - sebagian besar dataset sudah disertakan
2. Beberapa pelajaran mungkin memerlukan pengunduhan data - periksa README pelajaran
3. Pastikan Anda telah menarik perubahan terbaru:
   ```bash
   git pull origin main
   ```

---

## Pesan Kesalahan Umum

### Kesalahan Memori

**Kesalahan**: `MemoryError` atau kernel mati saat memproses data

**Solusi**:
```python
# Load data in chunks
for chunk in pd.read_csv('large_file.csv', chunksize=10000):
    process(chunk)

# Or read only needed columns
df = pd.read_csv('file.csv', usecols=['col1', 'col2'])

# Free memory when done
del large_dataframe
import gc
gc.collect()
```

### Peringatan Konvergensi

**Peringatan**: `ConvergenceWarning: Maximum number of iterations reached`

**Solusi**:
```python
from sklearn.linear_model import LogisticRegression

# Increase max iterations
model = LogisticRegression(max_iter=1000)

# Or scale your features first
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### Masalah Plotting

**Masalah**: Plot tidak muncul di Jupyter

**Solusi**:
```python
# Enable inline plotting
%matplotlib inline

# Import pyplot
import matplotlib.pyplot as plt

# Show plot explicitly
plt.plot(data)
plt.show()
```

**Masalah**: Plot Seaborn terlihat berbeda atau menghasilkan kesalahan

**Solusi**:
```python
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Update to compatible version
# pip install --upgrade seaborn matplotlib
```

### Kesalahan Unicode/Encoding

**Masalah**: `UnicodeDecodeError` saat membaca file

**Solusi**:
```python
# Specify encoding explicitly
df = pd.read_csv('file.csv', encoding='utf-8')

# Or try different encoding
df = pd.read_csv('file.csv', encoding='latin-1')

# For errors='ignore' to skip problematic characters
df = pd.read_csv('file.csv', encoding='utf-8', errors='ignore')
```

---

## Masalah Performa

### Eksekusi Notebook Lambat

**Masalah**: Notebook sangat lambat dijalankan

**Solusi**:
1. **Restart kernel untuk membebaskan memori**: `Kernel → Restart`
2. **Tutup notebook yang tidak digunakan** untuk membebaskan sumber daya
3. **Gunakan sampel data yang lebih kecil untuk pengujian**:
   ```python
   # Work with subset during development
   df_sample = df.sample(n=1000)
   ```
4. **Profil kode Anda** untuk menemukan hambatan:
   ```python
   %time operation()  # Time single operation
   %timeit operation()  # Time with multiple runs
   ```

### Penggunaan Memori Tinggi

**Masalah**: Sistem kehabisan memori

**Solusi**:
```python
# Check memory usage
df.info(memory_usage='deep')

# Optimize data types
df['column'] = df['column'].astype('int32')  # Instead of int64

# Drop unnecessary columns
df = df[['col1', 'col2']]  # Keep only needed columns

# Process in batches
for batch in np.array_split(df, 10):
    process(batch)
```

---

## Lingkungan dan Konfigurasi

### Masalah Lingkungan Virtual

**Masalah**: Lingkungan virtual tidak aktif

**Solusi**:
```bash
# Windows
python -m venv venv
venv\Scripts\activate.bat

# macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Check if activated (should show venv name in prompt)
which python  # Should point to venv python
```

**Masalah**: Paket terinstal tetapi tidak ditemukan di notebook

**Solusi**:
```bash
# Ensure notebook uses the correct kernel
# Install ipykernel in your venv
pip install ipykernel
python -m ipykernel install --user --name=ml-env --display-name="Python (ml-env)"

# In Jupyter: Kernel → Change Kernel → Python (ml-env)
```

### Masalah Git

**Masalah**: Tidak dapat menarik perubahan terbaru - konflik merge

**Solusi**:
```bash
# Stash your changes
git stash

# Pull latest
git pull origin main

# Reapply your changes
git stash pop

# If conflicts, resolve manually or:
git checkout --theirs path/to/file  # Take remote version
git checkout --ours path/to/file    # Keep your version
```

### Integrasi VS Code

**Masalah**: Notebook Jupyter tidak dapat dibuka di VS Code

**Solusi**:
1. Instal ekstensi Python di VS Code
2. Instal ekstensi Jupyter di VS Code
3. Pilih interpreter Python yang benar: `Ctrl+Shift+P` → "Python: Select Interpreter"
4. Restart VS Code

---

## Sumber Daya Tambahan

- **Diskusi Discord**: [Ajukan pertanyaan dan bagikan solusi di channel #ml-for-beginners](https://aka.ms/foundry/discord)
- **Microsoft Learn**: [Modul ML untuk Pemula](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)
- **Tutorial Video**: [Playlist YouTube](https://aka.ms/ml-beginners-videos)
- **Pelacak Masalah**: [Laporkan bug](https://github.com/microsoft/ML-For-Beginners/issues)

---

## Masih Mengalami Masalah?

Jika Anda telah mencoba solusi di atas dan masih mengalami masalah:

1. **Cari masalah yang sudah ada**: [GitHub Issues](https://github.com/microsoft/ML-For-Beginners/issues)
2. **Periksa diskusi di Discord**: [Diskusi Discord](https://aka.ms/foundry/discord)
3. **Buka masalah baru**: Sertakan:
   - Sistem operasi dan versinya
   - Versi Python/R
   - Pesan kesalahan (traceback lengkap)
   - Langkah-langkah untuk mereproduksi masalah
   - Apa yang sudah Anda coba

Kami siap membantu! 🚀

---

**Penafian**:  
Dokumen ini telah diterjemahkan menggunakan layanan penerjemahan AI [Co-op Translator](https://github.com/Azure/co-op-translator). Meskipun kami berusaha untuk memberikan hasil yang akurat, harap diperhatikan bahwa terjemahan otomatis mungkin mengandung kesalahan atau ketidakakuratan. Dokumen asli dalam bahasa aslinya harus dianggap sebagai sumber yang otoritatif. Untuk informasi yang bersifat kritis, disarankan menggunakan jasa penerjemahan manusia profesional. Kami tidak bertanggung jawab atas kesalahpahaman atau interpretasi yang keliru yang timbul dari penggunaan terjemahan ini.