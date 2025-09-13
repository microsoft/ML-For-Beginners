<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "9836ff53cfef716ddfd70e06c5f43436",
  "translation_date": "2025-09-03T17:53:39+00:00",
  "source_file": "3-Web-App/README.md",
  "language_code": "zh"
}
-->
# 构建一个使用您的机器学习模型的网页应用

在本课程的这一部分，您将学习一个应用型的机器学习主题：如何将您的 Scikit-learn 模型保存为一个文件，以便在网页应用中进行预测。一旦模型保存完成，您将学习如何在使用 Flask 构建的网页应用中使用它。您将首先使用一些关于 UFO 目击事件的数据创建一个模型！然后，您将构建一个网页应用，允许用户输入持续时间（秒数）、纬度和经度值，以预测哪个国家报告了看到 UFO。

![UFO 停车场](../../../translated_images/ufo.9e787f5161da9d4d1dafc537e1da09be8210f2ee996cb638aa5cee1d92867a04.zh.jpg)

照片由 <a href="https://unsplash.com/@mdherren?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Michael Herren</a> 提供，来自 <a href="https://unsplash.com/s/photos/ufo?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>

## 课程

1. [构建一个网页应用](1-Web-App/README.md)

## 致谢

“构建一个网页应用”由 [Jen Looper](https://twitter.com/jenlooper) 倾情撰写。

♥️ 测验由 Rohan Raj 编写。

数据集来源于 [Kaggle](https://www.kaggle.com/NUFORC/ufo-sightings)。

网页应用架构部分参考了 [这篇文章](https://towardsdatascience.com/how-to-easily-deploy-machine-learning-models-using-flask-b95af8fe34d4) 和 [这个仓库](https://github.com/abhinavsagar/machine-learning-deployment)，作者为 Abhinav Sagar。

---

**免责声明**：  
本文档使用AI翻译服务[Co-op Translator](https://github.com/Azure/co-op-translator)进行翻译。尽管我们努力确保翻译的准确性，但请注意，自动翻译可能包含错误或不准确之处。原始语言的文档应被视为权威来源。对于关键信息，建议使用专业人工翻译。我们对因使用此翻译而产生的任何误解或误读不承担责任。