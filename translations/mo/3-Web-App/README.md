<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "9836ff53cfef716ddfd70e06c5f43436",
  "translation_date": "2025-08-29T21:34:33+00:00",
  "source_file": "3-Web-App/README.md",
  "language_code": "mo"
}
-->
# 建立一個使用您的機器學習模型的網頁應用程式

在本課程的這一部分，您將學習一個應用機器學習的主題：如何將您的 Scikit-learn 模型保存為一個文件，並在網頁應用程式中使用它進行預測。一旦模型保存完成，您將學習如何在使用 Flask 建立的網頁應用程式中使用它。首先，您將使用一些關於 UFO 目擊事件的數據來建立模型！接著，您將建立一個網頁應用程式，允許您輸入秒數、緯度和經度值，來預測哪個國家報告了看到 UFO。

![UFO 停車場](../../../translated_images/ufo.9e787f5161da9d4d1dafc537e1da09be8210f2ee996cb638aa5cee1d92867a04.mo.jpg)

照片由 <a href="https://unsplash.com/@mdherren?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Michael Herren</a> 提供，來自 <a href="https://unsplash.com/s/photos/ufo?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>

## 課程

1. [建立網頁應用程式](1-Web-App/README.md)

## 致謝

"建立網頁應用程式" 由 [Jen Looper](https://twitter.com/jenlooper) 用 ♥️ 撰寫。

♥️ 測驗由 Rohan Raj 撰寫。

數據集來源於 [Kaggle](https://www.kaggle.com/NUFORC/ufo-sightings)。

網頁應用程式架構部分參考了 [這篇文章](https://towardsdatascience.com/how-to-easily-deploy-machine-learning-models-using-flask-b95af8fe34d4) 和 [這個倉庫](https://github.com/abhinavsagar/machine-learning-deployment)，由 Abhinav Sagar 提供。

---

**免責聲明**：  
本文件使用 AI 翻譯服務 [Co-op Translator](https://github.com/Azure/co-op-translator) 進行翻譯。我們致力於提供準確的翻譯，但請注意，自動翻譯可能包含錯誤或不準確之處。應以原始語言的文件作為權威來源。對於關鍵資訊，建議尋求專業人工翻譯。我們對於因使用本翻譯而引起的任何誤解或錯誤解讀概不負責。  