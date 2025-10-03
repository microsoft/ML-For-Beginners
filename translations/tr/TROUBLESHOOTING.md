<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "134d8759f0e2ab886e9aa4f62362c201",
  "translation_date": "2025-10-03T12:45:47+00:00",
  "source_file": "TROUBLESHOOTING.md",
  "language_code": "tr"
}
-->
# Sorun Giderme Kılavuzu

Bu kılavuz, Machine Learning for Beginners müfredatıyla çalışırken karşılaşabileceğiniz yaygın sorunları çözmenize yardımcı olur. Burada bir çözüm bulamazsanız, [Discord Tartışmaları](https://aka.ms/foundry/discord) bölümüne göz atabilir veya [bir sorun bildirebilirsiniz](https://github.com/microsoft/ML-For-Beginners/issues).

## İçindekiler

- [Kurulum Sorunları](../..)
- [Jupyter Notebook Sorunları](../..)
- [Python Paket Sorunları](../..)
- [R Ortamı Sorunları](../..)
- [Quiz Uygulaması Sorunları](../..)
- [Veri ve Dosya Yolu Sorunları](../..)
- [Yaygın Hata Mesajları](../..)
- [Performans Sorunları](../..)
- [Ortam ve Yapılandırma](../..)

---

## Kurulum Sorunları

### Python Kurulumu

**Sorun**: `python: komut bulunamadı`

**Çözüm**:
1. [python.org](https://www.python.org/downloads/) adresinden Python 3.8 veya daha yüksek bir sürümü yükleyin.
2. Kurulumu doğrulayın: `python --version` veya `python3 --version`
3. macOS/Linux'ta `python` yerine `python3` kullanmanız gerekebilir.

**Sorun**: Birden fazla Python sürümü çakışmalara neden oluyor

**Çözüm**:
```bash
# Use virtual environments to isolate projects
python -m venv ml-env

# Activate virtual environment
# On Windows:
ml-env\Scripts\activate
# On macOS/Linux:
source ml-env/bin/activate
```

### Jupyter Kurulumu

**Sorun**: `jupyter: komut bulunamadı`

**Çözüm**:
```bash
# Install Jupyter
pip install jupyter

# Or with pip3
pip3 install jupyter

# Verify installation
jupyter --version
```

**Sorun**: Jupyter tarayıcıda açılmıyor

**Çözüm**:
```bash
# Try specifying the browser
jupyter notebook --browser=chrome

# Or copy the URL with token from terminal and paste in browser manually
# Look for: http://localhost:8888/?token=...
```

### R Kurulumu

**Sorun**: R paketleri yüklenmiyor

**Çözüm**:
```r
# Ensure you have the latest R version
# Install packages with dependencies
install.packages(c("tidyverse", "tidymodels", "caret"), dependencies = TRUE)

# If compilation fails, try installing binary versions
install.packages("package-name", type = "binary")
```

**Sorun**: IRkernel Jupyter'de mevcut değil

**Çözüm**:
```r
# In R console
install.packages('IRkernel')
IRkernel::installspec(user = TRUE)
```

---

## Jupyter Notebook Sorunları

### Kernel Sorunları

**Sorun**: Kernel sürekli çöküyor veya yeniden başlıyor

**Çözüm**:
1. Kernel'i yeniden başlatın: `Kernel → Restart`
2. Çıktıyı temizleyip yeniden başlatın: `Kernel → Restart & Clear Output`
3. Bellek sorunlarını kontrol edin (bkz. [Performans Sorunları](../..))
4. Sorunlu kodu belirlemek için hücreleri tek tek çalıştırmayı deneyin.

**Sorun**: Yanlış Python kernel'i seçilmiş

**Çözüm**:
1. Mevcut kernel'i kontrol edin: `Kernel → Change Kernel`
2. Doğru Python sürümünü seçin.
3. Kernel eksikse, oluşturun:
```bash
python -m ipykernel install --user --name=ml-env
```

**Sorun**: Kernel başlamıyor

**Çözüm**:
```bash
# Reinstall ipykernel
pip uninstall ipykernel
pip install ipykernel

# Register the kernel again
python -m ipykernel install --user
```

### Notebook Hücre Sorunları

**Sorun**: Hücreler çalışıyor ama çıktı göstermiyor

**Çözüm**:
1. Hücrenin hala çalışıp çalışmadığını kontrol edin (`[*]` göstergesine bakın).
2. Kernel'i yeniden başlatın ve tüm hücreleri çalıştırın: `Kernel → Restart & Run All`
3. Tarayıcı konsolunda JavaScript hatalarını kontrol edin (F12).

**Sorun**: Hücreler çalıştırılamıyor - "Çalıştır" düğmesine tıklayınca tepki yok

**Çözüm**:
1. Jupyter sunucusunun terminalde hala çalışıp çalışmadığını kontrol edin.
2. Tarayıcı sayfasını yenileyin.
3. Notebook'u kapatıp yeniden açın.
4. Jupyter sunucusunu yeniden başlatın.

---

## Python Paket Sorunları

### İçe Aktarma Hataları

**Sorun**: `ModuleNotFoundError: No module named 'sklearn'`

**Çözüm**:
```bash
pip install scikit-learn

# Common ML packages for this course
pip install scikit-learn pandas numpy matplotlib seaborn
```

**Sorun**: `ImportError: cannot import name 'X' from 'sklearn'`

**Çözüm**:
```bash
# Update scikit-learn to latest version
pip install --upgrade scikit-learn

# Check version
python -c "import sklearn; print(sklearn.__version__)"
```

### Sürüm Çakışmaları

**Sorun**: Paket sürümü uyumsuzluk hataları

**Çözüm**:
```bash
# Create a new virtual environment
python -m venv fresh-env
source fresh-env/bin/activate  # or fresh-env\Scripts\activate on Windows

# Install packages fresh
pip install jupyter scikit-learn pandas numpy matplotlib seaborn

# If specific version needed
pip install scikit-learn==1.3.0
```

**Sorun**: `pip install` izin hatalarıyla başarısız oluyor

**Çözüm**:
```bash
# Install for current user only
pip install --user package-name

# Or use virtual environment (recommended)
python -m venv venv
source venv/bin/activate
pip install package-name
```

### Veri Yükleme Sorunları

**Sorun**: CSV dosyalarını yüklerken `FileNotFoundError`

**Çözüm**:
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

## R Ortamı Sorunları

### Paket Kurulumu

**Sorun**: Paket kurulumu derleme hatalarıyla başarısız oluyor

**Çözüm**:
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

**Sorun**: `tidyverse` yüklenmiyor

**Çözüm**:
```r
# Install dependencies first
install.packages(c("rlang", "vctrs", "pillar"))

# Then install tidyverse
install.packages("tidyverse")

# Or install components individually
install.packages(c("dplyr", "ggplot2", "tidyr", "readr"))
```

### RMarkdown Sorunları

**Sorun**: RMarkdown render edilmiyor

**Çözüm**:
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

## Quiz Uygulaması Sorunları

### Derleme ve Kurulum

**Sorun**: `npm install` başarısız oluyor

**Çözüm**:
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

**Sorun**: 8080 portu zaten kullanılıyor

**Çözüm**:
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

### Derleme Hataları

**Sorun**: `npm run build` başarısız oluyor

**Çözüm**:
```bash
# Check Node.js version (should be 14+)
node --version

# Update Node.js if needed
# Then clean install
rm -rf node_modules package-lock.json
npm install
npm run build
```

**Sorun**: Linting hataları derlemeyi engelliyor

**Çözüm**:
```bash
# Fix auto-fixable issues
npm run lint -- --fix

# Or temporarily disable linting in build
# (not recommended for production)
```

---

## Veri ve Dosya Yolu Sorunları

### Yol Problemleri

**Sorun**: Notebook çalıştırılırken veri dosyaları bulunamıyor

**Çözüm**:
1. **Notebook'ları her zaman bulundukları dizinden çalıştırın**
   ```bash
   cd /path/to/lesson/folder
   jupyter notebook
   ```

2. **Kodda göreceli yolları kontrol edin**
   ```python
   # Correct path from notebook location
   df = pd.read_csv('../data/filename.csv')
   
   # Not from your terminal location
   ```

3. **Gerekirse mutlak yollar kullanın**
   ```python
   import os
   base_path = os.path.dirname(os.path.abspath(__file__))
   data_path = os.path.join(base_path, 'data', 'filename.csv')
   ```

### Eksik Veri Dosyaları

**Sorun**: Veri seti dosyaları eksik

**Çözüm**:
1. Verilerin depo içinde olması gerektiğini kontrol edin - çoğu veri seti dahil edilmiştir.
2. Bazı dersler veri indirmeyi gerektirebilir - ders README dosyasını kontrol edin.
3. En son değişiklikleri çektiğinizden emin olun:
   ```bash
   git pull origin main
   ```

---

## Yaygın Hata Mesajları

### Bellek Hataları

**Hata**: `MemoryError` veya kernel veri işlerken çöküyor

**Çözüm**:
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

### Yakınsama Uyarıları

**Uyarı**: `ConvergenceWarning: Maximum number of iterations reached`

**Çözüm**:
```python
from sklearn.linear_model import LogisticRegression

# Increase max iterations
model = LogisticRegression(max_iter=1000)

# Or scale your features first
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### Grafik Sorunları

**Sorun**: Grafikler Jupyter'de görünmüyor

**Çözüm**:
```python
# Enable inline plotting
%matplotlib inline

# Import pyplot
import matplotlib.pyplot as plt

# Show plot explicitly
plt.plot(data)
plt.show()
```

**Sorun**: Seaborn grafikler farklı görünüyor veya hata veriyor

**Çözüm**:
```python
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Update to compatible version
# pip install --upgrade seaborn matplotlib
```

### Unicode/Kodlama Hataları

**Sorun**: Dosyaları okurken `UnicodeDecodeError`

**Çözüm**:
```python
# Specify encoding explicitly
df = pd.read_csv('file.csv', encoding='utf-8')

# Or try different encoding
df = pd.read_csv('file.csv', encoding='latin-1')

# For errors='ignore' to skip problematic characters
df = pd.read_csv('file.csv', encoding='utf-8', errors='ignore')
```

---

## Performans Sorunları

### Yavaş Notebook Çalıştırma

**Sorun**: Notebook'lar çok yavaş çalışıyor

**Çözüm**:
1. **Belleği boşaltmak için kernel'i yeniden başlatın**: `Kernel → Restart`
2. **Kullanılmayan notebook'ları kapatın**: Kaynakları boşaltmak için.
3. **Test için daha küçük veri örnekleri kullanın**:
   ```python
   # Work with subset during development
   df_sample = df.sample(n=1000)
   ```
4. **Kodunuzu profil oluşturun**: Darboğazları bulmak için.
   ```python
   %time operation()  # Time single operation
   %timeit operation()  # Time with multiple runs
   ```

### Yüksek Bellek Kullanımı

**Sorun**: Sistem belleği tükeniyor

**Çözüm**:
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

## Ortam ve Yapılandırma

### Sanal Ortam Sorunları

**Sorun**: Sanal ortam etkinleştirilemiyor

**Çözüm**:
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

**Sorun**: Paketler yüklü ama notebook'ta bulunamıyor

**Çözüm**:
```bash
# Ensure notebook uses the correct kernel
# Install ipykernel in your venv
pip install ipykernel
python -m ipykernel install --user --name=ml-env --display-name="Python (ml-env)"

# In Jupyter: Kernel → Change Kernel → Python (ml-env)
```

### Git Sorunları

**Sorun**: En son değişiklikler çekilemiyor - birleştirme çakışmaları

**Çözüm**:
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

### VS Code Entegrasyonu

**Sorun**: Jupyter notebook'ları VS Code'da açılmıyor

**Çözüm**:
1. VS Code'da Python uzantısını yükleyin.
2. VS Code'da Jupyter uzantısını yükleyin.
3. Doğru Python yorumlayıcısını seçin: `Ctrl+Shift+P` → "Python: Select Interpreter"
4. VS Code'u yeniden başlatın.

---

## Ek Kaynaklar

- **Discord Tartışmaları**: [#ml-for-beginners kanalında sorular sorun ve çözümleri paylaşın](https://aka.ms/foundry/discord)
- **Microsoft Learn**: [ML for Beginners modülleri](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)
- **Video Eğitimleri**: [YouTube Oynatma Listesi](https://aka.ms/ml-beginners-videos)
- **Sorun Takipçisi**: [Hataları bildirin](https://github.com/microsoft/ML-For-Beginners/issues)

---

## Hala Sorun Yaşıyor musunuz?

Yukarıdaki çözümleri denediyseniz ve hala sorun yaşıyorsanız:

1. **Mevcut sorunları arayın**: [GitHub Issues](https://github.com/microsoft/ML-For-Beginners/issues)
2. **Discord'daki tartışmaları kontrol edin**: [Discord Tartışmaları](https://aka.ms/foundry/discord)
3. **Yeni bir sorun açın**: Şunları ekleyin:
   - İşletim sistemi ve sürümü
   - Python/R sürümü
   - Hata mesajı (tam traceback)
   - Sorunu yeniden oluşturmak için adımlar
   - Daha önce denedikleriniz

Yardımcı olmak için buradayız! 🚀

---

**Feragatname**:  
Bu belge, AI çeviri hizmeti [Co-op Translator](https://github.com/Azure/co-op-translator) kullanılarak çevrilmiştir. Doğruluk için çaba göstersek de, otomatik çevirilerin hata veya yanlışlıklar içerebileceğini lütfen unutmayın. Belgenin orijinal dili, yetkili kaynak olarak kabul edilmelidir. Kritik bilgiler için profesyonel insan çevirisi önerilir. Bu çevirinin kullanımından kaynaklanan yanlış anlamalar veya yanlış yorumlamalar için sorumluluk kabul etmiyoruz.