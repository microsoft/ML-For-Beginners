<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "93fdaa0fd38836e50c4793e2f2f25e8b",
  "translation_date": "2025-10-03T11:02:30+00:00",
  "source_file": "AGENTS.md",
  "language_code": "ko"
}
-->
# AGENTS.md

## 프로젝트 개요

이 프로젝트는 **초보자를 위한 머신 러닝**으로, Python(주로 Scikit-learn)과 R을 사용하여 고전적인 머신 러닝 개념을 다루는 12주, 26강의 종합 커리큘럼입니다. 이 저장소는 자율 학습 리소스로 설계되었으며, 실습 프로젝트, 퀴즈, 과제가 포함되어 있습니다. 각 강의는 전 세계 다양한 문화와 지역의 실제 데이터를 통해 머신 러닝 개념을 탐구합니다.

주요 구성 요소:
- **교육 콘텐츠**: 머신 러닝 소개, 회귀, 분류, 클러스터링, NLP, 시계열, 강화 학습을 다루는 26개의 강의
- **퀴즈 애플리케이션**: Vue.js 기반 퀴즈 앱으로 강의 전후 평가 제공
- **다국어 지원**: GitHub Actions를 통해 40개 이상의 언어로 자동 번역
- **이중 언어 지원**: Python(Jupyter 노트북)과 R(R Markdown 파일)로 제공되는 강의
- **프로젝트 기반 학습**: 각 주제에 실습 프로젝트와 과제 포함

## 저장소 구조

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

각 강의 폴더는 일반적으로 다음을 포함합니다:
- `README.md` - 주요 강의 내용
- `notebook.ipynb` - Python Jupyter 노트북
- `solution/` - 솔루션 코드(Python 및 R 버전)
- `assignment.md` - 연습 문제
- `images/` - 시각 자료

## 설정 명령

### Python 강의용

대부분의 강의는 Jupyter 노트북을 사용합니다. 필요한 종속성을 설치하세요:

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

### R 강의용

R 강의는 `solution/R/` 폴더에 `.rmd` 또는 `.ipynb` 파일로 제공됩니다:

```bash
# Install R and required packages
# In R console:
install.packages(c("tidyverse", "tidymodels", "caret"))
```

### 퀴즈 애플리케이션

퀴즈 앱은 `quiz-app/` 디렉토리에 위치한 Vue.js 애플리케이션입니다:

```bash
cd quiz-app
npm install
```

### 문서 사이트용

문서를 로컬에서 실행하려면:

```bash
# Install Docsify
npm install -g docsify-cli

# Serve from repository root
docsify serve

# Access at http://localhost:3000
```

## 개발 워크플로우

### 강의 노트북 작업

1. 강의 디렉토리로 이동 (예: `2-Regression/1-Tools/`)
2. Jupyter 노트북 열기:
   ```bash
   jupyter notebook notebook.ipynb
   ```
3. 강의 내용과 연습 문제 진행
4. 필요 시 `solution/` 폴더에서 솔루션 확인

### Python 개발

- 강의는 표준 Python 데이터 과학 라이브러리를 사용
- Jupyter 노트북을 통한 대화형 학습
- 각 강의의 `solution/` 폴더에 솔루션 코드 제공

### R 개발

- R 강의는 `.rmd` 형식(R Markdown)으로 제공
- 솔루션은 `solution/R/` 하위 디렉토리에 위치
- RStudio 또는 R 커널이 포함된 Jupyter를 사용하여 R 노트북 실행

### 퀴즈 애플리케이션 개발

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

## 테스트 지침

### 퀴즈 애플리케이션 테스트

```bash
cd quiz-app

# Lint code
npm run lint

# Build to verify no errors
npm run build
```

**참고**: 이 저장소는 주로 교육용 커리큘럼입니다. 강의 내용에 대한 자동화된 테스트는 없습니다. 검증은 다음을 통해 이루어집니다:
- 강의 연습 문제 완료
- 노트북 셀을 성공적으로 실행
- 솔루션에서 예상 결과 확인

## 코드 스타일 지침

### Python 코드
- PEP 8 스타일 지침 준수
- 명확하고 설명적인 변수 이름 사용
- 복잡한 작업에 대한 주석 포함
- Jupyter 노트북에는 개념을 설명하는 마크다운 셀 포함

### JavaScript/Vue.js (퀴즈 앱)
- Vue.js 스타일 가이드 준수
- `quiz-app/package.json`에 ESLint 구성
- `npm run lint` 실행하여 문제 확인 및 자동 수정

### 문서
- 마크다운 파일은 명확하고 잘 구조화되어야 함
- 코드 예제는 fenced code blocks에 포함
- 내부 참조는 상대 링크 사용
- 기존 형식 규칙 준수

## 빌드 및 배포

### 퀴즈 애플리케이션 배포

퀴즈 앱은 Azure Static Web Apps에 배포할 수 있습니다:

1. **필수 조건**:
   - Azure 계정
   - GitHub 저장소(이미 fork됨)

2. **Azure에 배포**:
   - Azure Static Web App 리소스 생성
   - GitHub 저장소 연결
   - 앱 위치 설정: `/quiz-app`
   - 출력 위치 설정: `dist`
   - Azure가 자동으로 GitHub Actions 워크플로우 생성

3. **GitHub Actions 워크플로우**:
   - `.github/workflows/azure-static-web-apps-*.yml`에 워크플로우 파일 생성
   - main 브랜치로 푸시 시 자동으로 빌드 및 배포

### 문서 PDF 생성

문서에서 PDF 생성:

```bash
npm install
npm run convert
```

## 번역 워크플로우

**중요**: 번역은 GitHub Actions를 통해 Co-op Translator를 사용하여 자동화됩니다.

- 번역은 `main` 브랜치에 변경 사항이 푸시될 때 자동 생성
- **콘텐츠를 수동으로 번역하지 마세요** - 시스템이 처리합니다
- 워크플로우는 `.github/workflows/co-op-translator.yml`에 정의
- Azure AI/OpenAI 서비스를 사용하여 번역
- 40개 이상의 언어 지원

## 기여 지침

### 콘텐츠 기여자용

1. **저장소를 fork**하고 기능 브랜치 생성
2. **강의 콘텐츠 변경** 시 추가/업데이트
3. **번역된 파일 수정 금지** - 자동 생성됨
4. **코드 테스트** - 모든 노트북 셀이 성공적으로 실행되도록 확인
5. **링크와 이미지**가 올바르게 작동하는지 확인
6. **명확한 설명과 함께 pull request 제출**

### Pull Request 지침

- **제목 형식**: `[섹션] 변경 사항 간단 설명`
  - 예: `[Regression] 강의 5의 오타 수정`
  - 예: `[Quiz-App] 종속성 업데이트`
- **제출 전**:
  - 모든 노트북 셀이 오류 없이 실행되는지 확인
  - 퀴즈 앱 수정 시 `npm run lint` 실행
  - 마크다운 형식 확인
  - 새로운 코드 예제 테스트
- **PR에 포함해야 할 내용**:
  - 변경 사항 설명
  - 변경 이유
  - UI 변경 시 스크린샷
- **행동 강령**: [Microsoft 오픈 소스 행동 강령](CODE_OF_CONDUCT.md) 준수
- **CLA**: Contributor License Agreement 서명 필요

## 강의 구조

각 강의는 일관된 패턴을 따릅니다:

1. **강의 전 퀴즈** - 기본 지식 테스트
2. **강의 내용** - 작성된 지침 및 설명
3. **코드 시연** - 노트북에서 실습 예제
4. **지식 확인** - 학습 이해도 확인
5. **도전 과제** - 개념을 독립적으로 적용
6. **과제** - 확장된 연습
7. **강의 후 퀴즈** - 학습 결과 평가

## 공통 명령 참조

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

## 추가 리소스

- **Microsoft Learn Collection**: [초보자를 위한 머신 러닝 모듈](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)
- **퀴즈 앱**: [온라인 퀴즈](https://ff-quizzes.netlify.app/en/ml/)
- **토론 게시판**: [GitHub Discussions](https://github.com/microsoft/ML-For-Beginners/discussions)
- **비디오 워크스루**: [YouTube 재생 목록](https://aka.ms/ml-beginners-videos)

## 주요 기술

- **Python**: 머신 러닝 강의의 주요 언어(Scikit-learn, Pandas, NumPy, Matplotlib)
- **R**: tidyverse, tidymodels, caret을 사용하는 대안 구현
- **Jupyter**: Python 강의를 위한 대화형 노트북
- **R Markdown**: R 강의를 위한 문서
- **Vue.js 3**: 퀴즈 애플리케이션 프레임워크
- **Flask**: 머신 러닝 모델 배포를 위한 웹 애플리케이션 프레임워크
- **Docsify**: 문서 사이트 생성기
- **GitHub Actions**: CI/CD 및 자동 번역

## 보안 고려 사항

- **코드에 비밀 정보 없음**: API 키나 자격 증명을 커밋하지 않음
- **종속성**: npm 및 pip 패키지 최신 상태 유지
- **사용자 입력**: Flask 웹 앱 예제는 기본 입력 유효성 검사 포함
- **민감한 데이터**: 예제 데이터셋은 공개적이고 민감하지 않음

## 문제 해결

### Jupyter 노트북

- **커널 문제**: 셀이 멈추면 커널 재시작: Kernel → Restart
- **Import 오류**: pip으로 필요한 패키지가 설치되었는지 확인
- **경로 문제**: 노트북이 포함된 디렉토리에서 실행

### 퀴즈 애플리케이션

- **npm install 실패**: npm 캐시 삭제: `npm cache clean --force`
- **포트 충돌**: 포트 변경: `npm run serve -- --port 8081`
- **빌드 오류**: `node_modules` 삭제 후 재설치: `rm -rf node_modules && npm install`

### R 강의

- **패키지 없음**: 설치 명령: `install.packages("package-name")`
- **RMarkdown 렌더링**: rmarkdown 패키지가 설치되었는지 확인
- **커널 문제**: Jupyter에서 IRkernel 설치 필요할 수 있음

## 프로젝트 관련 참고 사항

- 이 프로젝트는 주로 **학습 커리큘럼**이며, 프로덕션 코드가 아님
- **실습을 통한 머신 러닝 개념 이해**에 초점
- 코드 예제는 **최적화보다 명확성**을 우선
- 대부분의 강의는 **독립적**으로 완료 가능
- **솔루션 제공**되지만 학습자가 먼저 연습 문제를 시도해야 함
- 저장소는 **Docsify**를 사용하여 빌드 단계 없이 웹 문서 제공
- **스케치 노트**로 개념의 시각적 요약 제공
- **다국어 지원**으로 콘텐츠를 글로벌하게 접근 가능

---

**면책 조항**:  
이 문서는 AI 번역 서비스 [Co-op Translator](https://github.com/Azure/co-op-translator)를 사용하여 번역되었습니다. 정확성을 위해 최선을 다하고 있으나, 자동 번역에는 오류나 부정확성이 포함될 수 있습니다. 원본 문서의 원어 버전을 신뢰할 수 있는 권위 있는 자료로 간주해야 합니다. 중요한 정보의 경우, 전문적인 인간 번역을 권장합니다. 이 번역 사용으로 인해 발생하는 오해나 잘못된 해석에 대해 당사는 책임을 지지 않습니다.