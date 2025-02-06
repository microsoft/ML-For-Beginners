# 小测验

这些小测验是机器学习课程的课前和课后测验，课程网址为 https://aka.ms/ml-beginners

## 项目设置

```
npm install
```

### 编译和热重载用于开发

```
npm run serve
```

### 编译和压缩用于生产环境

```
npm run build
```

### 代码检查和修复文件

```
npm run lint
```

### 自定义配置

请参阅[配置参考](https://cli.vuejs.org/config/)。

致谢：感谢此测验应用的原版：https://github.com/arpan45/simple-quiz-vue

## 部署到 Azure

以下是帮助你入门的分步指南：

1. Fork 一个 GitHub 仓库
确保你的静态网页应用代码在你的 GitHub 仓库中。Fork 此仓库。

2. 创建一个 Azure 静态网页应用
- 创建一个 [Azure 账号](http://azure.microsoft.com)
- 访问 [Azure 门户](https://portal.azure.com)
- 点击“创建资源”，搜索“静态网页应用”。
- 点击“创建”。

3. 配置静态网页应用
- 基本信息：订阅：选择你的 Azure 订阅。
- 资源组：创建一个新的资源组或使用现有的资源组。
- 名称：为你的静态网页应用提供一个名称。
- 区域：选择离你的用户最近的区域。

- #### 部署详情：
- 源：选择“GitHub”。
- GitHub 账号：授权 Azure 访问你的 GitHub 账号。
- 组织：选择你的 GitHub 组织。
- 仓库：选择包含你的静态网页应用的仓库。
- 分支：选择你想要部署的分支。

- #### 构建详情：
- 构建预设：选择你的应用使用的框架（例如，React、Angular、Vue 等）。
- 应用位置：指定包含你应用代码的文件夹（例如，如果在根目录则为 /）。
- API 位置：如果你有 API，指定其位置（可选）。
- 输出位置：指定生成构建输出的文件夹（例如，build 或 dist）。

4. 审核并创建
审核你的设置并点击“创建”。Azure 将设置必要的资源并在你的仓库中创建一个 GitHub Actions 工作流程。

5. GitHub Actions 工作流程
Azure 将自动在你的仓库中创建一个 GitHub Actions 工作流程文件 (.github/workflows/azure-static-web-apps-<name>.yml)。此工作流程将处理构建和部署过程。

6. 监控部署
前往你的 GitHub 仓库中的“Actions”标签。
你应该会看到一个工作流程正在运行。此工作流程将构建并部署你的静态网页应用到 Azure。
一旦工作流程完成，你的应用将在提供的 Azure URL 上上线。

### 示例工作流程文件

以下是 GitHub Actions 工作流程文件的示例：
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

### 其他资源
- [Azure 静态网页应用文档](https://learn.microsoft.com/azure/static-web-apps/getting-started)
- [GitHub Actions 文档](https://docs.github.com/actions/use-cases-and-examples/deploying/deploying-to-azure-static-web-app)

**免责声明**：
本文档已使用基于机器的人工智能翻译服务进行翻译。尽管我们努力确保准确性，但请注意，自动翻译可能包含错误或不准确之处。应将原始语言的文档视为权威来源。对于关键信息，建议使用专业的人类翻译。对于因使用此翻译而产生的任何误解或误读，我们概不负责。