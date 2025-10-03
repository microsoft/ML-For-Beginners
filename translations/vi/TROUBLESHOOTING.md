<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "134d8759f0e2ab886e9aa4f62362c201",
  "translation_date": "2025-10-03T12:50:15+00:00",
  "source_file": "TROUBLESHOOTING.md",
  "language_code": "vi"
}
-->
# Hướng Dẫn Khắc Phục Sự Cố

Hướng dẫn này giúp bạn giải quyết các vấn đề thường gặp khi làm việc với chương trình học Machine Learning for Beginners. Nếu bạn không tìm thấy giải pháp ở đây, hãy kiểm tra [Thảo luận trên Discord](https://aka.ms/foundry/discord) hoặc [mở một vấn đề mới](https://github.com/microsoft/ML-For-Beginners/issues).

## Mục Lục

- [Vấn Đề Cài Đặt](../..)
- [Vấn Đề Jupyter Notebook](../..)
- [Vấn Đề Gói Python](../..)
- [Vấn Đề Môi Trường R](../..)
- [Vấn Đề Ứng Dụng Quiz](../..)
- [Vấn Đề Dữ Liệu và Đường Dẫn Tệp](../..)
- [Thông Báo Lỗi Thường Gặp](../..)
- [Vấn Đề Hiệu Suất](../..)
- [Môi Trường và Cấu Hình](../..)

---

## Vấn Đề Cài Đặt

### Cài Đặt Python

**Vấn Đề**: `python: command not found`

**Giải Pháp**:
1. Cài đặt Python 3.8 hoặc cao hơn từ [python.org](https://www.python.org/downloads/)
2. Kiểm tra cài đặt: `python --version` hoặc `python3 --version`
3. Trên macOS/Linux, bạn có thể cần sử dụng `python3` thay vì `python`

**Vấn Đề**: Nhiều phiên bản Python gây xung đột

**Giải Pháp**:
```bash
# Use virtual environments to isolate projects
python -m venv ml-env

# Activate virtual environment
# On Windows:
ml-env\Scripts\activate
# On macOS/Linux:
source ml-env/bin/activate
```

### Cài Đặt Jupyter

**Vấn Đề**: `jupyter: command not found`

**Giải Pháp**:
```bash
# Install Jupyter
pip install jupyter

# Or with pip3
pip3 install jupyter

# Verify installation
jupyter --version
```

**Vấn Đề**: Jupyter không mở được trong trình duyệt

**Giải Pháp**:
```bash
# Try specifying the browser
jupyter notebook --browser=chrome

# Or copy the URL with token from terminal and paste in browser manually
# Look for: http://localhost:8888/?token=...
```

### Cài Đặt R

**Vấn Đề**: Không cài đặt được các gói R

**Giải Pháp**:
```r
# Ensure you have the latest R version
# Install packages with dependencies
install.packages(c("tidyverse", "tidymodels", "caret"), dependencies = TRUE)

# If compilation fails, try installing binary versions
install.packages("package-name", type = "binary")
```

**Vấn Đề**: IRkernel không khả dụng trong Jupyter

**Giải Pháp**:
```r
# In R console
install.packages('IRkernel')
IRkernel::installspec(user = TRUE)
```

---

## Vấn Đề Jupyter Notebook

### Vấn Đề Kernel

**Vấn Đề**: Kernel liên tục bị dừng hoặc khởi động lại

**Giải Pháp**:
1. Khởi động lại kernel: `Kernel → Restart`
2. Xóa đầu ra và khởi động lại: `Kernel → Restart & Clear Output`
3. Kiểm tra vấn đề bộ nhớ (xem [Vấn Đề Hiệu Suất](../..))
4. Thử chạy từng ô để xác định đoạn mã gây lỗi

**Vấn Đề**: Chọn sai kernel Python

**Giải Pháp**:
1. Kiểm tra kernel hiện tại: `Kernel → Change Kernel`
2. Chọn phiên bản Python đúng
3. Nếu kernel bị thiếu, tạo kernel mới:
```bash
python -m ipykernel install --user --name=ml-env
```

**Vấn Đề**: Kernel không khởi động được

**Giải Pháp**:
```bash
# Reinstall ipykernel
pip uninstall ipykernel
pip install ipykernel

# Register the kernel again
python -m ipykernel install --user
```

### Vấn Đề Ô Notebook

**Vấn Đề**: Ô đang chạy nhưng không hiển thị đầu ra

**Giải Pháp**:
1. Kiểm tra xem ô có đang chạy không (tìm chỉ báo `[*]`)
2. Khởi động lại kernel và chạy tất cả các ô: `Kernel → Restart & Run All`
3. Kiểm tra bảng điều khiển trình duyệt để tìm lỗi JavaScript (F12)

**Vấn Đề**: Không thể chạy ô - không có phản hồi khi nhấp "Run"

**Giải Pháp**:
1. Kiểm tra xem máy chủ Jupyter có đang chạy trong terminal không
2. Làm mới trang trình duyệt
3. Đóng và mở lại notebook
4. Khởi động lại máy chủ Jupyter

---

## Vấn Đề Gói Python

### Lỗi Import

**Vấn Đề**: `ModuleNotFoundError: No module named 'sklearn'`

**Giải Pháp**:
```bash
pip install scikit-learn

# Common ML packages for this course
pip install scikit-learn pandas numpy matplotlib seaborn
```

**Vấn Đề**: `ImportError: cannot import name 'X' from 'sklearn'`

**Giải Pháp**:
```bash
# Update scikit-learn to latest version
pip install --upgrade scikit-learn

# Check version
python -c "import sklearn; print(sklearn.__version__)"
```

### Xung Đột Phiên Bản

**Vấn Đề**: Lỗi không tương thích phiên bản gói

**Giải Pháp**:
```bash
# Create a new virtual environment
python -m venv fresh-env
source fresh-env/bin/activate  # or fresh-env\Scripts\activate on Windows

# Install packages fresh
pip install jupyter scikit-learn pandas numpy matplotlib seaborn

# If specific version needed
pip install scikit-learn==1.3.0
```

**Vấn Đề**: `pip install` không thành công do lỗi quyền

**Giải Pháp**:
```bash
# Install for current user only
pip install --user package-name

# Or use virtual environment (recommended)
python -m venv venv
source venv/bin/activate
pip install package-name
```

### Vấn Đề Tải Dữ Liệu

**Vấn Đề**: `FileNotFoundError` khi tải tệp CSV

**Giải Pháp**:
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

## Vấn Đề Môi Trường R

### Cài Đặt Gói

**Vấn Đề**: Cài đặt gói thất bại với lỗi biên dịch

**Giải Pháp**:
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

**Vấn Đề**: Không cài đặt được `tidyverse`

**Giải Pháp**:
```r
# Install dependencies first
install.packages(c("rlang", "vctrs", "pillar"))

# Then install tidyverse
install.packages("tidyverse")

# Or install components individually
install.packages(c("dplyr", "ggplot2", "tidyr", "readr"))
```

### Vấn Đề RMarkdown

**Vấn Đề**: RMarkdown không hiển thị

**Giải Pháp**:
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

## Vấn Đề Ứng Dụng Quiz

### Xây Dựng và Cài Đặt

**Vấn Đề**: `npm install` không thành công

**Giải Pháp**:
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

**Vấn Đề**: Cổng 8080 đã được sử dụng

**Giải Pháp**:
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

### Lỗi Xây Dựng

**Vấn Đề**: `npm run build` không thành công

**Giải Pháp**:
```bash
# Check Node.js version (should be 14+)
node --version

# Update Node.js if needed
# Then clean install
rm -rf node_modules package-lock.json
npm install
npm run build
```

**Vấn Đề**: Lỗi linting ngăn cản việc xây dựng

**Giải Pháp**:
```bash
# Fix auto-fixable issues
npm run lint -- --fix

# Or temporarily disable linting in build
# (not recommended for production)
```

---

## Vấn Đề Dữ Liệu và Đường Dẫn Tệp

### Vấn Đề Đường Dẫn

**Vấn Đề**: Không tìm thấy tệp dữ liệu khi chạy notebook

**Giải Pháp**:
1. **Luôn chạy notebook từ thư mục chứa nó**
   ```bash
   cd /path/to/lesson/folder
   jupyter notebook
   ```

2. **Kiểm tra đường dẫn tương đối trong mã**
   ```python
   # Correct path from notebook location
   df = pd.read_csv('../data/filename.csv')
   
   # Not from your terminal location
   ```

3. **Sử dụng đường dẫn tuyệt đối nếu cần**
   ```python
   import os
   base_path = os.path.dirname(os.path.abspath(__file__))
   data_path = os.path.join(base_path, 'data', 'filename.csv')
   ```

### Thiếu Tệp Dữ Liệu

**Vấn Đề**: Các tệp dữ liệu bị thiếu

**Giải Pháp**:
1. Kiểm tra xem dữ liệu có nên nằm trong kho lưu trữ không - hầu hết các bộ dữ liệu đều được bao gồm
2. Một số bài học có thể yêu cầu tải dữ liệu - kiểm tra README của bài học
3. Đảm bảo bạn đã kéo các thay đổi mới nhất:
   ```bash
   git pull origin main
   ```

---

## Thông Báo Lỗi Thường Gặp

### Lỗi Bộ Nhớ

**Lỗi**: `MemoryError` hoặc kernel bị dừng khi xử lý dữ liệu

**Giải Pháp**:
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

### Cảnh Báo Hội Tụ

**Cảnh Báo**: `ConvergenceWarning: Maximum number of iterations reached`

**Giải Pháp**:
```python
from sklearn.linear_model import LogisticRegression

# Increase max iterations
model = LogisticRegression(max_iter=1000)

# Or scale your features first
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### Vấn Đề Vẽ Biểu Đồ

**Vấn Đề**: Biểu đồ không hiển thị trong Jupyter

**Giải Pháp**:
```python
# Enable inline plotting
%matplotlib inline

# Import pyplot
import matplotlib.pyplot as plt

# Show plot explicitly
plt.plot(data)
plt.show()
```

**Vấn Đề**: Biểu đồ Seaborn hiển thị khác hoặc gây lỗi

**Giải Pháp**:
```python
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Update to compatible version
# pip install --upgrade seaborn matplotlib
```

### Lỗi Unicode/Mã Hóa

**Vấn Đề**: `UnicodeDecodeError` khi đọc tệp

**Giải Pháp**:
```python
# Specify encoding explicitly
df = pd.read_csv('file.csv', encoding='utf-8')

# Or try different encoding
df = pd.read_csv('file.csv', encoding='latin-1')

# For errors='ignore' to skip problematic characters
df = pd.read_csv('file.csv', encoding='utf-8', errors='ignore')
```

---

## Vấn Đề Hiệu Suất

### Notebook Chạy Chậm

**Vấn Đề**: Notebook chạy rất chậm

**Giải Pháp**:
1. **Khởi động lại kernel để giải phóng bộ nhớ**: `Kernel → Restart`
2. **Đóng các notebook không sử dụng** để giải phóng tài nguyên
3. **Sử dụng mẫu dữ liệu nhỏ hơn để thử nghiệm**:
   ```python
   # Work with subset during development
   df_sample = df.sample(n=1000)
   ```
4. **Phân tích mã của bạn** để tìm điểm nghẽn:
   ```python
   %time operation()  # Time single operation
   %timeit operation()  # Time with multiple runs
   ```

### Sử Dụng Bộ Nhớ Cao

**Vấn Đề**: Hệ thống hết bộ nhớ

**Giải Pháp**:
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

## Môi Trường và Cấu Hình

### Vấn Đề Môi Trường Ảo

**Vấn Đề**: Môi trường ảo không kích hoạt được

**Giải Pháp**:
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

**Vấn Đề**: Các gói đã cài đặt nhưng không tìm thấy trong notebook

**Giải Pháp**:
```bash
# Ensure notebook uses the correct kernel
# Install ipykernel in your venv
pip install ipykernel
python -m ipykernel install --user --name=ml-env --display-name="Python (ml-env)"

# In Jupyter: Kernel → Change Kernel → Python (ml-env)
```

### Vấn Đề Git

**Vấn Đề**: Không thể kéo các thay đổi mới nhất - xung đột hợp nhất

**Giải Pháp**:
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

### Tích Hợp VS Code

**Vấn Đề**: Jupyter notebook không mở được trong VS Code

**Giải Pháp**:
1. Cài đặt tiện ích Python trong VS Code
2. Cài đặt tiện ích Jupyter trong VS Code
3. Chọn trình thông dịch Python đúng: `Ctrl+Shift+P` → "Python: Select Interpreter"
4. Khởi động lại VS Code

---

## Tài Nguyên Bổ Sung

- **Thảo luận trên Discord**: [Đặt câu hỏi và chia sẻ giải pháp trong kênh #ml-for-beginners](https://aka.ms/foundry/discord)
- **Microsoft Learn**: [Các module ML for Beginners](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)
- **Video Hướng Dẫn**: [Danh sách phát trên YouTube](https://aka.ms/ml-beginners-videos)
- **Theo dõi vấn đề**: [Báo cáo lỗi](https://github.com/microsoft/ML-For-Beginners/issues)

---

## Vẫn Gặp Vấn Đề?

Nếu bạn đã thử các giải pháp trên mà vẫn gặp vấn đề:

1. **Tìm kiếm các vấn đề hiện có**: [GitHub Issues](https://github.com/microsoft/ML-For-Beginners/issues)
2. **Kiểm tra thảo luận trên Discord**: [Thảo luận trên Discord](https://aka.ms/foundry/discord)
3. **Mở một vấn đề mới**: Bao gồm:
   - Hệ điều hành và phiên bản của bạn
   - Phiên bản Python/R
   - Thông báo lỗi (toàn bộ traceback)
   - Các bước để tái hiện vấn đề
   - Những gì bạn đã thử

Chúng tôi luôn sẵn sàng giúp đỡ! 🚀

---

**Tuyên bố miễn trừ trách nhiệm**:  
Tài liệu này đã được dịch bằng dịch vụ dịch thuật AI [Co-op Translator](https://github.com/Azure/co-op-translator). Mặc dù chúng tôi cố gắng đảm bảo độ chính xác, xin lưu ý rằng các bản dịch tự động có thể chứa lỗi hoặc không chính xác. Tài liệu gốc bằng ngôn ngữ bản địa nên được coi là nguồn thông tin chính thức. Đối với các thông tin quan trọng, khuyến nghị sử dụng dịch vụ dịch thuật chuyên nghiệp bởi con người. Chúng tôi không chịu trách nhiệm cho bất kỳ sự hiểu lầm hoặc diễn giải sai nào phát sinh từ việc sử dụng bản dịch này.