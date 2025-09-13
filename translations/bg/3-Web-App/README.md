<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "9836ff53cfef716ddfd70e06c5f43436",
  "translation_date": "2025-09-05T00:36:12+00:00",
  "source_file": "3-Web-App/README.md",
  "language_code": "bg"
}
-->
# Създайте уеб приложение за използване на вашия ML модел

В тази част от учебната програма ще се запознаете с приложна тема в машинното обучение: как да запазите вашия Scikit-learn модел като файл, който може да се използва за правене на прогнози в рамките на уеб приложение. След като моделът бъде запазен, ще научите как да го използвате в уеб приложение, създадено с Flask. Първо ще създадете модел, използвайки данни, свързани с наблюдения на НЛО! След това ще изградите уеб приложение, което ще ви позволи да въведете брой секунди, заедно със стойности за географска ширина и дължина, за да предвидите коя държава е докладвала за наблюдение на НЛО.

![Паркинг за НЛО](../../../3-Web-App/images/ufo.jpg)

Снимка от <a href="https://unsplash.com/@mdherren?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Michael Herren</a> на <a href="https://unsplash.com/s/photos/ufo?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>

## Уроци

1. [Създайте уеб приложение](1-Web-App/README.md)

## Кредити

"Създайте уеб приложение" е написано с ♥️ от [Jen Looper](https://twitter.com/jenlooper).

♥️ Тестовете са написани от Rohan Raj.

Данните са взети от [Kaggle](https://www.kaggle.com/NUFORC/ufo-sightings).

Архитектурата на уеб приложението е частично предложена от [тази статия](https://towardsdatascience.com/how-to-easily-deploy-machine-learning-models-using-flask-b95af8fe34d4) и [този репозиторий](https://github.com/abhinavsagar/machine-learning-deployment) от Abhinav Sagar.

---

**Отказ от отговорност**:  
Този документ е преведен с помощта на AI услуга за превод [Co-op Translator](https://github.com/Azure/co-op-translator). Въпреки че се стремим към точност, моля, имайте предвид, че автоматизираните преводи може да съдържат грешки или неточности. Оригиналният документ на неговия роден език трябва да се счита за авторитетен източник. За критична информация се препоръчва професионален човешки превод. Ние не носим отговорност за недоразумения или погрешни интерпретации, произтичащи от използването на този превод.