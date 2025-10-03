<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "134d8759f0e2ab886e9aa4f62362c201",
  "translation_date": "2025-10-03T12:40:34+00:00",
  "source_file": "TROUBLESHOOTING.md",
  "language_code": "ko"
}
-->
# 문제 해결 가이드

이 가이드는 Machine Learning for Beginners 커리큘럼을 사용할 때 발생하는 일반적인 문제를 해결하는 데 도움을 줍니다. 여기서 해결책을 찾지 못한 경우, [Discord Discussions](https://aka.ms/foundry/discord) 또는 [문제 등록](https://github.com/microsoft/ML-For-Beginners/issues)을 확인하세요.

## 목차

- [설치 문제](../..)
- [Jupyter Notebook 문제](../..)
- [Python 패키지 문제](../..)
- [R 환경 문제](../..)
- [퀴즈 애플리케이션 문제](../..)
- [데이터 및 파일 경로 문제](../..)
- [일반적인 오류 메시지](../..)
- [성능 문제](../..)
- [환경 및 설정](../..)

---

## 설치 문제

### Python 설치

**문제**: `python: command not found`

**해결책**:
1. [python.org](https://www.python.org/downloads/)에서 Python 3.8 이상을 설치하세요.
2. 설치 확인: `python --version` 또는 `python3 --version`
3. macOS/Linux에서는 `python` 대신 `python3`을 사용해야 할 수 있습니다.

**문제**: 여러 Python 버전이 충돌을 일으킴

**해결책**:
```bash
# Use virtual environments to isolate projects
python -m venv ml-env

# Activate virtual environment
# On Windows:
ml-env\Scripts\activate
# On macOS/Linux:
source ml-env/bin/activate
```

### Jupyter 설치

**문제**: `jupyter: command not found`

**해결책**:
```bash
# Install Jupyter
pip install jupyter

# Or with pip3
pip3 install jupyter

# Verify installation
jupyter --version
```

**문제**: Jupyter가 브라우저에서 실행되지 않음

**해결책**:
```bash
# Try specifying the browser
jupyter notebook --browser=chrome

# Or copy the URL with token from terminal and paste in browser manually
# Look for: http://localhost:8888/?token=...
```

### R 설치

**문제**: R 패키지가 설치되지 않음

**해결책**:
```r
# Ensure you have the latest R version
# Install packages with dependencies
install.packages(c("tidyverse", "tidymodels", "caret"), dependencies = TRUE)

# If compilation fails, try installing binary versions
install.packages("package-name", type = "binary")
```

**문제**: IRkernel이 Jupyter에서 사용 불가

**해결책**:
```r
# In R console
install.packages('IRkernel')
IRkernel::installspec(user = TRUE)
```

---

## Jupyter Notebook 문제

### 커널 문제

**문제**: 커널이 계속 죽거나 재시작됨

**해결책**:
1. 커널 재시작: `Kernel → Restart`
2. 출력 지우고 재시작: `Kernel → Restart & Clear Output`
3. 메모리 문제 확인 ([성능 문제](../..) 참조)
4. 개별 셀을 실행하여 문제 코드 식별

**문제**: 잘못된 Python 커널 선택됨

**해결책**:
1. 현재 커널 확인: `Kernel → Change Kernel`
2. 올바른 Python 버전 선택
3. 커널이 없으면 생성:
```bash
python -m ipykernel install --user --name=ml-env
```

**문제**: 커널이 시작되지 않음

**해결책**:
```bash
# Reinstall ipykernel
pip uninstall ipykernel
pip install ipykernel

# Register the kernel again
python -m ipykernel install --user
```

### Notebook 셀 문제

**문제**: 셀이 실행되지만 출력이 표시되지 않음

**해결책**:
1. 셀이 실행 중인지 확인 (`[*]` 표시 확인)
2. 커널 재시작 후 모든 셀 실행: `Kernel → Restart & Run All`
3. 브라우저 콘솔에서 JavaScript 오류 확인 (F12)

**문제**: 셀 실행 불가 - "Run" 클릭 시 반응 없음

**해결책**:
1. Jupyter 서버가 터미널에서 실행 중인지 확인
2. 브라우저 페이지 새로고침
3. Notebook 닫고 다시 열기
4. Jupyter 서버 재시작

---

## Python 패키지 문제

### Import 오류

**문제**: `ModuleNotFoundError: No module named 'sklearn'`

**해결책**:
```bash
pip install scikit-learn

# Common ML packages for this course
pip install scikit-learn pandas numpy matplotlib seaborn
```

**문제**: `ImportError: cannot import name 'X' from 'sklearn'`

**해결책**:
```bash
# Update scikit-learn to latest version
pip install --upgrade scikit-learn

# Check version
python -c "import sklearn; print(sklearn.__version__)"
```

### 버전 충돌

**문제**: 패키지 버전 호환성 오류

**해결책**:
```bash
# Create a new virtual environment
python -m venv fresh-env
source fresh-env/bin/activate  # or fresh-env\Scripts\activate on Windows

# Install packages fresh
pip install jupyter scikit-learn pandas numpy matplotlib seaborn

# If specific version needed
pip install scikit-learn==1.3.0
```

**문제**: `pip install`이 권한 오류로 실패

**해결책**:
```bash
# Install for current user only
pip install --user package-name

# Or use virtual environment (recommended)
python -m venv venv
source venv/bin/activate
pip install package-name
```

### 데이터 로딩 문제

**문제**: CSV 파일 로딩 시 `FileNotFoundError`

**해결책**:
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

## R 환경 문제

### 패키지 설치

**문제**: 패키지 설치가 컴파일 오류로 실패

**해결책**:
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

**문제**: `tidyverse` 설치 불가

**해결책**:
```r
# Install dependencies first
install.packages(c("rlang", "vctrs", "pillar"))

# Then install tidyverse
install.packages("tidyverse")

# Or install components individually
install.packages(c("dplyr", "ggplot2", "tidyr", "readr"))
```

### RMarkdown 문제

**문제**: RMarkdown이 렌더링되지 않음

**해결책**:
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

## 퀴즈 애플리케이션 문제

### 빌드 및 설치

**문제**: `npm install` 실패

**해결책**:
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

**문제**: 포트 8080이 이미 사용 중

**해결책**:
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

### 빌드 오류

**문제**: `npm run build` 실패

**해결책**:
```bash
# Check Node.js version (should be 14+)
node --version

# Update Node.js if needed
# Then clean install
rm -rf node_modules package-lock.json
npm install
npm run build
```

**문제**: 린팅 오류로 빌드 불가

**해결책**:
```bash
# Fix auto-fixable issues
npm run lint -- --fix

# Or temporarily disable linting in build
# (not recommended for production)
```

---

## 데이터 및 파일 경로 문제

### 경로 문제

**문제**: Notebook 실행 시 데이터 파일을 찾을 수 없음

**해결책**:
1. **항상 Notebook을 포함된 디렉토리에서 실행하세요**
   ```bash
   cd /path/to/lesson/folder
   jupyter notebook
   ```

2. **코드에서 상대 경로 확인**
   ```python
   # Correct path from notebook location
   df = pd.read_csv('../data/filename.csv')
   
   # Not from your terminal location
   ```

3. **필요 시 절대 경로 사용**
   ```python
   import os
   base_path = os.path.dirname(os.path.abspath(__file__))
   data_path = os.path.join(base_path, 'data', 'filename.csv')
   ```

### 데이터 파일 누락

**문제**: 데이터셋 파일이 누락됨

**해결책**:
1. 데이터가 저장소에 포함되어야 하는지 확인 - 대부분의 데이터셋은 포함되어 있음
2. 일부 레슨은 데이터 다운로드가 필요할 수 있음 - 레슨 README 확인
3. 최신 변경 사항을 가져왔는지 확인:
   ```bash
   git pull origin main
   ```

---

## 일반적인 오류 메시지

### 메모리 오류

**오류**: `MemoryError` 또는 데이터 처리 중 커널 종료

**해결책**:
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

### 수렴 경고

**경고**: `ConvergenceWarning: Maximum number of iterations reached`

**해결책**:
```python
from sklearn.linear_model import LogisticRegression

# Increase max iterations
model = LogisticRegression(max_iter=1000)

# Or scale your features first
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### 플로팅 문제

**문제**: Jupyter에서 플롯이 표시되지 않음

**해결책**:
```python
# Enable inline plotting
%matplotlib inline

# Import pyplot
import matplotlib.pyplot as plt

# Show plot explicitly
plt.plot(data)
plt.show()
```

**문제**: Seaborn 플롯이 다르게 보이거나 오류 발생

**해결책**:
```python
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Update to compatible version
# pip install --upgrade seaborn matplotlib
```

### 유니코드/인코딩 오류

**문제**: 파일 읽기 시 `UnicodeDecodeError`

**해결책**:
```python
# Specify encoding explicitly
df = pd.read_csv('file.csv', encoding='utf-8')

# Or try different encoding
df = pd.read_csv('file.csv', encoding='latin-1')

# For errors='ignore' to skip problematic characters
df = pd.read_csv('file.csv', encoding='utf-8', errors='ignore')
```

---

## 성능 문제

### Notebook 실행 속도 저하

**문제**: Notebook 실행이 매우 느림

**해결책**:
1. **메모리 확보를 위해 커널 재시작**: `Kernel → Restart`
2. **사용하지 않는 Notebook 닫기**로 리소스 확보
3. **테스트용으로 작은 데이터 샘플 사용**:
   ```python
   # Work with subset during development
   df_sample = df.sample(n=1000)
   ```
4. **코드 프로파일링**으로 병목 현상 찾기:
   ```python
   %time operation()  # Time single operation
   %timeit operation()  # Time with multiple runs
   ```

### 높은 메모리 사용량

**문제**: 시스템 메모리 부족

**해결책**:
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

## 환경 및 설정

### 가상 환경 문제

**문제**: 가상 환경이 활성화되지 않음

**해결책**:
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

**문제**: 패키지가 설치되었지만 Notebook에서 찾을 수 없음

**해결책**:
```bash
# Ensure notebook uses the correct kernel
# Install ipykernel in your venv
pip install ipykernel
python -m ipykernel install --user --name=ml-env --display-name="Python (ml-env)"

# In Jupyter: Kernel → Change Kernel → Python (ml-env)
```

### Git 문제

**문제**: 최신 변경 사항을 가져올 수 없음 - 병합 충돌

**해결책**:
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

### VS Code 통합

**문제**: Jupyter Notebook이 VS Code에서 열리지 않음

**해결책**:
1. VS Code에서 Python 확장 설치
2. VS Code에서 Jupyter 확장 설치
3. 올바른 Python 인터프리터 선택: `Ctrl+Shift+P` → "Python: Select Interpreter"
4. VS Code 재시작

---

## 추가 자료

- **Discord Discussions**: [#ml-for-beginners 채널에서 질문하고 해결책 공유](https://aka.ms/foundry/discord)
- **Microsoft Learn**: [ML for Beginners 모듈](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)
- **비디오 튜토리얼**: [YouTube 재생 목록](https://aka.ms/ml-beginners-videos)
- **문제 추적기**: [버그 신고](https://github.com/microsoft/ML-For-Beginners/issues)

---

## 여전히 문제가 있나요?

위의 해결책을 시도했지만 여전히 문제가 발생한다면:

1. **기존 문제 검색**: [GitHub Issues](https://github.com/microsoft/ML-For-Beginners/issues)
2. **Discord에서 논의 확인**: [Discord Discussions](https://aka.ms/foundry/discord)
3. **새로운 문제 등록**: 다음을 포함하세요:
   - 운영 체제 및 버전
   - Python/R 버전
   - 오류 메시지 (전체 추적)
   - 문제를 재현하는 단계
   - 이미 시도한 해결책

우리는 도와드릴 준비가 되어 있습니다! 🚀

---

**면책 조항**:  
이 문서는 AI 번역 서비스 [Co-op Translator](https://github.com/Azure/co-op-translator)를 사용하여 번역되었습니다. 정확성을 위해 최선을 다하고 있으나, 자동 번역에는 오류나 부정확성이 포함될 수 있습니다. 원본 문서의 원어 버전이 권위 있는 출처로 간주되어야 합니다. 중요한 정보의 경우, 전문적인 인간 번역을 권장합니다. 이 번역 사용으로 인해 발생하는 오해나 잘못된 해석에 대해 당사는 책임을 지지 않습니다.