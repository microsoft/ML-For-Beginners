# 通过翻译课程做出贡献

我们欢迎对本课程中的课程进行翻译！
## 指南

每个课程文件夹和课程介绍文件夹中都有包含翻译后 markdown 文件的文件夹。

> 注意，请不要翻译代码示例文件中的任何代码；唯一需要翻译的是 README、作业和测验。谢谢！

翻译后的文件应遵循以下命名规范：

**README._[language]_.md**

其中 _[language]_ 是遵循 ISO 639-1 标准的两字母语言缩写（例如 `README.es.md` 表示西班牙语，`README.nl.md` 表示荷兰语）。

**assignment._[language]_.md**

与 README 类似，请也翻译作业。

> 重要提示：在翻译此仓库中的文本时，请确保不使用机器翻译。我们将通过社区验证翻译，因此请仅在您精通的语言中自愿翻译。

**测验**

1. 通过在此处添加文件将您的翻译添加到 quiz-app：https://github.com/microsoft/ML-For-Beginners/tree/main/quiz-app/src/assets/translations，文件命名规范为（en.json, fr.json）。**但请不要本地化单词 'true' 或 'false'。谢谢！**

2. 将您的语言代码添加到 quiz-app 的 App.vue 文件中的下拉菜单中。

3. 编辑 quiz-app 的 [translations index.js 文件](https://github.com/microsoft/ML-For-Beginners/blob/main/quiz-app/src/assets/translations/index.js) 以添加您的语言。

4. 最后，编辑您翻译后的 README.md 文件中的所有测验链接，以直接指向您的翻译测验：https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/1 变为 https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/1?loc=id

**感谢您**

我们真诚地感谢您的努力！

**免责声明**:
本文档使用基于机器的人工智能翻译服务进行翻译。尽管我们力求准确，但请注意，自动翻译可能包含错误或不准确之处。应将原始语言的文档视为权威来源。对于关键信息，建议使用专业的人类翻译。对于因使用此翻译而引起的任何误解或误读，我们不承担任何责任。