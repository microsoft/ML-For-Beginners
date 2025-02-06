# 퀴즈

이 퀴즈들은 https://aka.ms/ml-beginners 에 있는 ML 커리큘럼의 강의 전후 퀴즈입니다.

## 프로젝트 설정

```
npm install
```

### 개발을 위한 컴파일 및 핫 리로드

```
npm run serve
```

### 프로덕션을 위한 컴파일 및 최소화

```
npm run build
```

### 파일 린트 및 수정

```
npm run lint
```

### 구성 사용자 정의

[Configuration Reference](https://cli.vuejs.org/config/)를 참조하세요.

크레딧: 원본 퀴즈 앱 버전에 감사드립니다: https://github.com/arpan45/simple-quiz-vue

## Azure에 배포하기

다음은 시작하는 데 도움이 되는 단계별 가이드입니다:

1. GitHub 저장소 포크하기
정적 웹 앱 코드가 GitHub 저장소에 있는지 확인하세요. 이 저장소를 포크하세요.

2. Azure 정적 웹 앱 생성
- [Azure 계정](http://azure.microsoft.com)을 생성하세요.
- [Azure 포털](https://portal.azure.com)에 접속하세요.
- "리소스 생성"을 클릭하고 "Static Web App"을 검색하세요.
- "생성"을 클릭하세요.

3. 정적 웹 앱 구성
- 기본 설정: 구독: Azure 구독을 선택하세요.
- 리소스 그룹: 새 리소스 그룹을 생성하거나 기존 리소스 그룹을 사용하세요.
- 이름: 정적 웹 앱의 이름을 입력하세요.
- 지역: 사용자와 가장 가까운 지역을 선택하세요.

- #### 배포 세부 사항:
- 소스: "GitHub"를 선택하세요.
- GitHub 계정: Azure가 GitHub 계정에 접근할 수 있도록 권한을 부여하세요.
- 조직: GitHub 조직을 선택하세요.
- 저장소: 정적 웹 앱이 포함된 저장소를 선택하세요.
- 브랜치: 배포할 브랜치를 선택하세요.

- #### 빌드 세부 사항:
- 빌드 프리셋: 앱이 구축된 프레임워크를 선택하세요 (예: React, Angular, Vue 등).
- 앱 위치: 앱 코드가 포함된 폴더를 지정하세요 (예: 루트에 있는 경우 /).
- API 위치: API가 있는 경우 위치를 지정하세요 (선택 사항).
- 출력 위치: 빌드 출력이 생성되는 폴더를 지정하세요 (예: build 또는 dist).

4. 검토 및 생성
설정을 검토하고 "생성"을 클릭하세요. Azure는 필요한 리소스를 설정하고 GitHub 저장소에 GitHub Actions 워크플로우를 생성합니다.

5. GitHub Actions 워크플로우
Azure는 자동으로 저장소에 GitHub Actions 워크플로우 파일을 생성합니다 (.github/workflows/azure-static-web-apps-<name>.yml). 이 워크플로우는 빌드 및 배포 프로세스를 처리합니다.

6. 배포 모니터링
GitHub 저장소의 "Actions" 탭으로 이동하세요.
워크플로우가 실행 중인 것을 볼 수 있습니다. 이 워크플로우는 정적 웹 앱을 Azure에 빌드하고 배포할 것입니다.
워크플로우가 완료되면 제공된 Azure URL에서 앱이 활성화됩니다.

### 예제 워크플로우 파일

다음은 GitHub Actions 워크플로우 파일의 예제입니다:
name: Azure Static Web Apps CI/CD
```
on:
  push:
    branches:
      - main
  pull_request:
    types: [opened, synchronize, reopened, closed]
    branches:
      - main

jobs:
  build_and_deploy_job:
    runs-on: ubuntu-latest
    name: Build and Deploy Job
    steps:
      - uses: actions/checkout@v2
      - name: Build And Deploy
        id: builddeploy
        uses: Azure/static-web-apps-deploy@v1
        with:
          azure_static_web_apps_api_token: ${{ secrets.AZURE_STATIC_WEB_APPS_API_TOKEN }}
          repo_token: ${{ secrets.GITHUB_TOKEN }}
          action: "upload"
          app_location: "/quiz-app" # App source code path
          api_location: ""API source code path optional
          output_location: "dist" #Built app content directory - optional
```

### 추가 리소스
- [Azure Static Web Apps Documentation](https://learn.microsoft.com/azure/static-web-apps/getting-started)
- [GitHub Actions Documentation](https://docs.github.com/actions/use-cases-and-examples/deploying/deploying-to-azure-static-web-app)

**면책 조항**:
이 문서는 기계 기반 AI 번역 서비스를 사용하여 번역되었습니다. 정확성을 위해 노력하고 있지만, 자동 번역에는 오류나 부정확성이 있을 수 있음을 유의하시기 바랍니다. 원본 문서의 원어가 권위 있는 출처로 간주되어야 합니다. 중요한 정보의 경우, 전문적인 인간 번역을 권장합니다. 이 번역 사용으로 인해 발생하는 오해나 잘못된 해석에 대해 당사는 책임을 지지 않습니다.