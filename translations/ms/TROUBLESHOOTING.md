<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "134d8759f0e2ab886e9aa4f62362c201",
  "translation_date": "2025-10-03T12:51:12+00:00",
  "source_file": "TROUBLESHOOTING.md",
  "language_code": "ms"
}
-->
# Panduan Penyelesaian Masalah

Panduan ini membantu anda menyelesaikan masalah biasa semasa menggunakan kurikulum Machine Learning for Beginners. Jika anda tidak menemui penyelesaian di sini, sila semak [Perbincangan Discord](https://aka.ms/foundry/discord) atau [buka isu](https://github.com/microsoft/ML-For-Beginners/issues).

## Kandungan

- [Masalah Pemasangan](../..)
- [Masalah Jupyter Notebook](../..)
- [Masalah Pakej Python](../..)
- [Masalah Persekitaran R](../..)
- [Masalah Aplikasi Kuiz](../..)
- [Masalah Data dan Laluan Fail](../..)
- [Mesej Ralat Biasa](../..)
- [Masalah Prestasi](../..)
- [Persekitaran dan Konfigurasi](../..)

---

## Masalah Pemasangan

### Pemasangan Python

**Masalah**: `python: command not found`

**Penyelesaian**:
1. Pasang Python 3.8 atau lebih tinggi dari [python.org](https://www.python.org/downloads/)
2. Sahkan pemasangan: `python --version` atau `python3 --version`
3. Pada macOS/Linux, anda mungkin perlu menggunakan `python3` dan bukannya `python`

**Masalah**: Versi Python berganda menyebabkan konflik

**Penyelesaian**:
```bash
# Use virtual environments to isolate projects
python -m venv ml-env

# Activate virtual environment
# On Windows:
ml-env\Scripts\activate
# On macOS/Linux:
source ml-env/bin/activate
```

### Pemasangan Jupyter

**Masalah**: `jupyter: command not found`

**Penyelesaian**:
```bash
# Install Jupyter
pip install jupyter

# Or with pip3
pip3 install jupyter

# Verify installation
jupyter --version
```

**Masalah**: Jupyter tidak dilancarkan dalam pelayar

**Penyelesaian**:
```bash
# Try specifying the browser
jupyter notebook --browser=chrome

# Or copy the URL with token from terminal and paste in browser manually
# Look for: http://localhost:8888/?token=...
```

### Pemasangan R

**Masalah**: Pakej R tidak dapat dipasang

**Penyelesaian**:
```r
# Ensure you have the latest R version
# Install packages with dependencies
install.packages(c("tidyverse", "tidymodels", "caret"), dependencies = TRUE)

# If compilation fails, try installing binary versions
install.packages("package-name", type = "binary")
```

**Masalah**: IRkernel tidak tersedia dalam Jupyter

**Penyelesaian**:
```r
# In R console
install.packages('IRkernel')
IRkernel::installspec(user = TRUE)
```

---

## Masalah Jupyter Notebook

### Masalah Kernel

**Masalah**: Kernel sering mati atau dimulakan semula

**Penyelesaian**:
1. Mulakan semula kernel: `Kernel → Restart`
2. Kosongkan output dan mulakan semula: `Kernel → Restart & Clear Output`
3. Periksa masalah memori (lihat [Masalah Prestasi](../..))
4. Cuba jalankan sel satu persatu untuk mengenal pasti kod yang bermasalah

**Masalah**: Kernel Python yang salah dipilih

**Penyelesaian**:
1. Periksa kernel semasa: `Kernel → Change Kernel`
2. Pilih versi Python yang betul
3. Jika kernel tiada, buat kernel:
```bash
python -m ipykernel install --user --name=ml-env
```

**Masalah**: Kernel tidak dapat dimulakan

**Penyelesaian**:
```bash
# Reinstall ipykernel
pip uninstall ipykernel
pip install ipykernel

# Register the kernel again
python -m ipykernel install --user
```

### Masalah Sel Notebook

**Masalah**: Sel sedang berjalan tetapi tidak menunjukkan output

**Penyelesaian**:
1. Periksa jika sel masih berjalan (cari penunjuk `[*]`)
2. Mulakan semula kernel dan jalankan semua sel: `Kernel → Restart & Run All`
3. Periksa konsol pelayar untuk ralat JavaScript (F12)

**Masalah**: Tidak dapat menjalankan sel - tiada respons apabila klik "Run"

**Penyelesaian**:
1. Periksa jika pelayan Jupyter masih berjalan dalam terminal
2. Segarkan semula halaman pelayar
3. Tutup dan buka semula notebook
4. Mulakan semula pelayan Jupyter

---

## Masalah Pakej Python

### Ralat Import

**Masalah**: `ModuleNotFoundError: No module named 'sklearn'`

**Penyelesaian**:
```bash
pip install scikit-learn

# Common ML packages for this course
pip install scikit-learn pandas numpy matplotlib seaborn
```

**Masalah**: `ImportError: cannot import name 'X' from 'sklearn'`

**Penyelesaian**:
```bash
# Update scikit-learn to latest version
pip install --upgrade scikit-learn

# Check version
python -c "import sklearn; print(sklearn.__version__)"
```

### Konflik Versi

**Masalah**: Ralat ketidakserasian versi pakej

**Penyelesaian**:
```bash
# Create a new virtual environment
python -m venv fresh-env
source fresh-env/bin/activate  # or fresh-env\Scripts\activate on Windows

# Install packages fresh
pip install jupyter scikit-learn pandas numpy matplotlib seaborn

# If specific version needed
pip install scikit-learn==1.3.0
```

**Masalah**: `pip install` gagal dengan ralat kebenaran

**Penyelesaian**:
```bash
# Install for current user only
pip install --user package-name

# Or use virtual environment (recommended)
python -m venv venv
source venv/bin/activate
pip install package-name
```

### Masalah Memuatkan Data

**Masalah**: `FileNotFoundError` semasa memuatkan fail CSV

**Penyelesaian**:
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

## Masalah Persekitaran R

### Pemasangan Pakej

**Masalah**: Pemasangan pakej gagal dengan ralat pengkompil

**Penyelesaian**:
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

**Masalah**: `tidyverse` tidak dapat dipasang

**Penyelesaian**:
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

**Penyelesaian**:
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

## Masalah Aplikasi Kuiz

### Pembinaan dan Pemasangan

**Masalah**: `npm install` gagal

**Penyelesaian**:
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

**Penyelesaian**:
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

### Ralat Pembinaan

**Masalah**: `npm run build` gagal

**Penyelesaian**:
```bash
# Check Node.js version (should be 14+)
node --version

# Update Node.js if needed
# Then clean install
rm -rf node_modules package-lock.json
npm install
npm run build
```

**Masalah**: Ralat linting menghalang pembinaan

**Penyelesaian**:
```bash
# Fix auto-fixable issues
npm run lint -- --fix

# Or temporarily disable linting in build
# (not recommended for production)
```

---

## Masalah Data dan Laluan Fail

### Masalah Laluan

**Masalah**: Fail data tidak ditemui semasa menjalankan notebook

**Penyelesaian**:
1. **Sentiasa jalankan notebook dari direktori yang mengandungi fail tersebut**
   ```bash
   cd /path/to/lesson/folder
   jupyter notebook
   ```

2. **Periksa laluan relatif dalam kod**
   ```python
   # Correct path from notebook location
   df = pd.read_csv('../data/filename.csv')
   
   # Not from your terminal location
   ```

3. **Gunakan laluan mutlak jika perlu**
   ```python
   import os
   base_path = os.path.dirname(os.path.abspath(__file__))
   data_path = os.path.join(base_path, 'data', 'filename.csv')
   ```

### Fail Data Hilang

**Masalah**: Fail dataset hilang

**Penyelesaian**:
1. Periksa jika data sepatutnya ada dalam repositori - kebanyakan dataset disertakan
2. Sesetengah pelajaran mungkin memerlukan muat turun data - semak README pelajaran
3. Pastikan anda telah menarik perubahan terkini:
   ```bash
   git pull origin main
   ```

---

## Mesej Ralat Biasa

### Ralat Memori

**Ralat**: `MemoryError` atau kernel mati semasa memproses data

**Penyelesaian**:
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

### Amaran Konvergensi

**Amaran**: `ConvergenceWarning: Maximum number of iterations reached`

**Penyelesaian**:
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

**Masalah**: Plot tidak muncul dalam Jupyter

**Penyelesaian**:
```python
# Enable inline plotting
%matplotlib inline

# Import pyplot
import matplotlib.pyplot as plt

# Show plot explicitly
plt.plot(data)
plt.show()
```

**Masalah**: Plot Seaborn kelihatan berbeza atau menghasilkan ralat

**Penyelesaian**:
```python
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Update to compatible version
# pip install --upgrade seaborn matplotlib
```

### Ralat Unicode/Pengekodan

**Masalah**: `UnicodeDecodeError` semasa membaca fail

**Penyelesaian**:
```python
# Specify encoding explicitly
df = pd.read_csv('file.csv', encoding='utf-8')

# Or try different encoding
df = pd.read_csv('file.csv', encoding='latin-1')

# For errors='ignore' to skip problematic characters
df = pd.read_csv('file.csv', encoding='utf-8', errors='ignore')
```

---

## Masalah Prestasi

### Pelaksanaan Notebook Perlahan

**Masalah**: Notebook sangat perlahan untuk dijalankan

**Penyelesaian**:
1. **Mulakan semula kernel untuk membebaskan memori**: `Kernel → Restart`
2. **Tutup notebook yang tidak digunakan** untuk membebaskan sumber
3. **Gunakan sampel data yang lebih kecil untuk ujian**:
   ```python
   # Work with subset during development
   df_sample = df.sample(n=1000)
   ```
4. **Profilkan kod anda** untuk mencari kelembapan:
   ```python
   %time operation()  # Time single operation
   %timeit operation()  # Time with multiple runs
   ```

### Penggunaan Memori Tinggi

**Masalah**: Sistem kehabisan memori

**Penyelesaian**:
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

## Persekitaran dan Konfigurasi

### Masalah Persekitaran Maya

**Masalah**: Persekitaran maya tidak diaktifkan

**Penyelesaian**:
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

**Masalah**: Pakej dipasang tetapi tidak ditemui dalam notebook

**Penyelesaian**:
```bash
# Ensure notebook uses the correct kernel
# Install ipykernel in your venv
pip install ipykernel
python -m ipykernel install --user --name=ml-env --display-name="Python (ml-env)"

# In Jupyter: Kernel → Change Kernel → Python (ml-env)
```

### Masalah Git

**Masalah**: Tidak dapat menarik perubahan terkini - konflik penggabungan

**Penyelesaian**:
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

**Masalah**: Notebook Jupyter tidak dapat dibuka dalam VS Code

**Penyelesaian**:
1. Pasang sambungan Python dalam VS Code
2. Pasang sambungan Jupyter dalam VS Code
3. Pilih interpreter Python yang betul: `Ctrl+Shift+P` → "Python: Select Interpreter"
4. Mulakan semula VS Code

---

## Sumber Tambahan

- **Perbincangan Discord**: [Ajukan soalan dan kongsi penyelesaian dalam saluran #ml-for-beginners](https://aka.ms/foundry/discord)
- **Microsoft Learn**: [Modul ML for Beginners](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)
- **Tutorial Video**: [Senarai Main YouTube](https://aka.ms/ml-beginners-videos)
- **Penjejak Isu**: [Laporkan pepijat](https://github.com/microsoft/ML-For-Beginners/issues)

---

## Masih Mengalami Masalah?

Jika anda telah mencuba penyelesaian di atas dan masih menghadapi masalah:

1. **Cari isu sedia ada**: [GitHub Issues](https://github.com/microsoft/ML-For-Beginners/issues)
2. **Semak perbincangan dalam Discord**: [Perbincangan Discord](https://aka.ms/foundry/discord)
3. **Buka isu baru**: Sertakan:
   - Sistem operasi dan versi anda
   - Versi Python/R
   - Mesej ralat (jejak penuh)
   - Langkah untuk menghasilkan semula masalah
   - Apa yang telah anda cuba

Kami sedia membantu! 🚀

---

**Penafian**:  
Dokumen ini telah diterjemahkan menggunakan perkhidmatan terjemahan AI [Co-op Translator](https://github.com/Azure/co-op-translator). Walaupun kami berusaha untuk memastikan ketepatan, sila ambil perhatian bahawa terjemahan automatik mungkin mengandungi kesilapan atau ketidaktepatan. Dokumen asal dalam bahasa asalnya harus dianggap sebagai sumber yang berwibawa. Untuk maklumat yang kritikal, terjemahan manusia profesional adalah disyorkan. Kami tidak bertanggungjawab atas sebarang salah faham atau salah tafsir yang timbul daripada penggunaan terjemahan ini.