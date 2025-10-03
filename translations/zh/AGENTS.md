<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "93fdaa0fd38836e50c4793e2f2f25e8b",
  "translation_date": "2025-10-03T10:59:57+00:00",
  "source_file": "AGENTS.md",
  "language_code": "zh"
}
-->
# AGENTS.md

## 项目概述

这是**机器学习入门**，一个全面的12周、26课的课程体系，涵盖使用Python（主要是Scikit-learn）和R的经典机器学习概念。该仓库设计为一个自学资源，包含实践项目、测验和作业。每节课通过来自世界各地不同文化和地区的真实数据探索机器学习概念。

关键组成部分：
- **教育内容**：26节课，涵盖机器学习简介、回归、分类、聚类、自然语言处理（NLP）、时间序列和强化学习
- **测验应用**：基于Vue.js的测验应用，提供课前和课后评估
- **多语言支持**：通过GitHub Actions自动翻译成40多种语言
- **双语言支持**：课程内容同时提供Python（Jupyter笔记本）和R（R Markdown文件）
- **基于项目的学习**：每个主题都包含实践项目和作业

## 仓库结构

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

每个课程文件夹通常包含：
- `README.md` - 主要课程内容
- `notebook.ipynb` - Python Jupyter笔记本
- `solution/` - 解决方案代码（Python和R版本）
- `assignment.md` - 练习题
- `images/` - 可视化资源

## 设置命令

### 针对Python课程

大多数课程使用Jupyter笔记本。安装所需依赖项：

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

### 针对R课程

R课程位于`solution/R/`文件夹中，以`.rmd`或`.ipynb`文件形式存在：

```bash
# Install R and required packages
# In R console:
install.packages(c("tidyverse", "tidymodels", "caret"))
```

### 针对测验应用

测验应用是一个位于`quiz-app/`目录中的Vue.js应用：

```bash
cd quiz-app
npm install
```

### 针对文档站点

本地运行文档：

```bash
# Install Docsify
npm install -g docsify-cli

# Serve from repository root
docsify serve

# Access at http://localhost:3000
```

## 开发工作流程

### 使用课程笔记本

1. 进入课程目录（例如，`2-Regression/1-Tools/`）
2. 打开Jupyter笔记本：
   ```bash
   jupyter notebook notebook.ipynb
   ```
3. 学习课程内容并完成练习
4. 如有需要，可查看`solution/`文件夹中的解决方案

### Python开发

- 课程使用标准的Python数据科学库
- Jupyter笔记本用于交互式学习
- 每节课的`solution/`文件夹中提供解决方案代码

### R开发

- R课程以`.rmd`格式（R Markdown）提供
- 解决方案位于`solution/R/`子目录中
- 使用RStudio或带有R内核的Jupyter运行R笔记本

### 测验应用开发

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

## 测试说明

### 测验应用测试

```bash
cd quiz-app

# Lint code
npm run lint

# Build to verify no errors
npm run build
```

**注意**：这是一个主要用于教育的课程仓库。课程内容没有自动化测试。验证通过以下方式完成：
- 完成课程练习
- 成功运行笔记本单元格
- 将输出与解决方案中的预期结果进行比较

## 代码风格指南

### Python代码
- 遵循PEP 8风格指南
- 使用清晰、描述性的变量名
- 对复杂操作添加注释
- Jupyter笔记本应包含解释概念的Markdown单元格

### JavaScript/Vue.js（测验应用）
- 遵循Vue.js风格指南
- ESLint配置位于`quiz-app/package.json`
- 运行`npm run lint`检查并自动修复问题

### 文档
- Markdown文件应清晰且结构良好
- 在代码块中包含代码示例
- 内部引用使用相对链接
- 遵循现有的格式约定

## 构建与部署

### 测验应用部署

测验应用可以部署到Azure静态Web应用：

1. **先决条件**：
   - Azure账户
   - GitHub仓库（已分叉）

2. **部署到Azure**：
   - 创建Azure静态Web应用资源
   - 连接到GitHub仓库
   - 设置应用位置：`/quiz-app`
   - 设置输出位置：`dist`
   - Azure会自动创建GitHub Actions工作流

3. **GitHub Actions工作流**：
   - 工作流文件创建于`.github/workflows/azure-static-web-apps-*.yml`
   - 推送到主分支时自动构建和部署

### 文档PDF

从文档生成PDF：

```bash
npm install
npm run convert
```

## 翻译工作流程

**重要**：翻译通过GitHub Actions使用Co-op Translator自动完成。

- 当更改推送到`main`分支时，翻译会自动生成
- **不要手动翻译内容** - 系统会处理
- 工作流定义在`.github/workflows/co-op-translator.yml`
- 使用Azure AI/OpenAI服务进行翻译
- 支持40多种语言

## 贡献指南

### 针对内容贡献者

1. **分叉仓库**并创建一个功能分支
2. **修改课程内容**以添加或更新课程
3. **不要修改翻译文件** - 它们是自动生成的
4. **测试代码** - 确保所有笔记本单元格成功运行
5. **验证链接和图片**是否正常工作
6. **提交拉取请求**并提供清晰的描述

### 拉取请求指南

- **标题格式**：`[部分] 简要描述更改`
  - 示例：`[回归] 修复第5课中的拼写错误`
  - 示例：`[测验应用] 更新依赖项`
- **提交前**：
  - 确保所有笔记本单元格无错误执行
  - 如果修改了测验应用，运行`npm run lint`
  - 验证Markdown格式
  - 测试任何新的代码示例
- **拉取请求必须包括**：
  - 更改描述
  - 更改原因
  - 如果有UI更改，提供截图
- **行为准则**：遵循[Microsoft开源行为准则](CODE_OF_CONDUCT.md)
- **CLA**：需要签署贡献者许可协议

## 课程结构

每节课遵循一致的模式：

1. **课前测验** - 测试基础知识
2. **课程内容** - 书面说明和解释
3. **代码演示** - 笔记本中的实践示例
4. **知识检查** - 验证学习理解
5. **挑战** - 独立应用概念
6. **作业** - 扩展练习
7. **课后测验** - 评估学习成果

## 常用命令参考

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

## 其他资源

- **Microsoft Learn集合**：[机器学习入门模块](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)
- **测验应用**：[在线测验](https://ff-quizzes.netlify.app/en/ml/)
- **讨论板**：[GitHub Discussions](https://github.com/microsoft/ML-For-Beginners/discussions)
- **视频讲解**：[YouTube播放列表](https://aka.ms/ml-beginners-videos)

## 关键技术

- **Python**：机器学习课程的主要语言（Scikit-learn, Pandas, NumPy, Matplotlib）
- **R**：使用tidyverse, tidymodels, caret的替代实现
- **Jupyter**：Python课程的交互式笔记本
- **R Markdown**：R课程的文档
- **Vue.js 3**：测验应用框架
- **Flask**：用于机器学习模型部署的Web应用框架
- **Docsify**：文档站点生成器
- **GitHub Actions**：CI/CD和自动翻译

## 安全注意事项

- **代码中不包含秘密信息**：不要提交API密钥或凭证
- **依赖项**：保持npm和pip包更新
- **用户输入**：Flask Web应用示例包括基本输入验证
- **敏感数据**：示例数据集是公开且无敏感信息的

## 故障排除

### Jupyter笔记本

- **内核问题**：如果单元格挂起，请重启内核：内核 → 重启
- **导入错误**：确保使用pip安装了所有必需的包
- **路径问题**：从笔记本所在目录运行笔记本

### 测验应用

- **npm安装失败**：清除npm缓存：`npm cache clean --force`
- **端口冲突**：更改端口：`npm run serve -- --port 8081`
- **构建错误**：删除`node_modules`并重新安装：`rm -rf node_modules && npm install`

### R课程

- **未找到包**：使用以下命令安装：`install.packages("package-name")`
- **RMarkdown渲染问题**：确保安装了rmarkdown包
- **内核问题**：可能需要为Jupyter安装IRkernel

## 项目特定说明

- 这主要是一个**学习课程**，而非生产代码
- 重点是通过实践练习**理解机器学习概念**
- 代码示例优先考虑**清晰性而非优化**
- 大多数课程是**独立的**，可以单独完成
- **提供解决方案**，但学习者应先尝试完成练习
- 仓库使用**Docsify**生成Web文档，无需构建步骤
- **手绘笔记**提供概念的可视化总结
- **多语言支持**使内容全球可访问

---

**免责声明**：  
本文档使用AI翻译服务 [Co-op Translator](https://github.com/Azure/co-op-translator) 进行翻译。尽管我们努力确保翻译的准确性，但请注意，自动翻译可能包含错误或不准确之处。原始语言的文档应被视为权威来源。对于关键信息，建议使用专业人工翻译。我们对因使用此翻译而产生的任何误解或误读不承担责任。