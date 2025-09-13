<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "b28a3a4911584062772c537b653ebbc7",
  "translation_date": "2025-09-05T12:08:20+00:00",
  "source_file": "5-Clustering/README.md",
  "language_code": "sr"
}
-->
# Модели кластерисања за машинско учење

Кластерисање је задатак машинског учења који има за циљ да пронађе објекте који личе један на други и групише их у групе које се називају кластери. Оно што кластерисање разликује од других приступа у машинском учењу је то што се ствари дешавају аутоматски; заправо, може се рећи да је то супротност надгледаном учењу.

## Регионална тема: модели кластерисања за музички укус публике у Нигерији 🎧

Разнолика публика у Нигерији има разнолике музичке укусе. Користећи податке прикупљене са Spotify-а (инспирисано [овим чланком](https://towardsdatascience.com/country-wise-visual-analysis-of-music-taste-using-spotify-api-seaborn-in-python-77f5b749b421)), погледајмо неке од популарних песама у Нигерији. Овај скуп података укључује информације о различитим песмама као што су оцена 'плесности', 'акустичности', јачина звука, 'говорност', популарност и енергија. Биће занимљиво открити обрасце у овим подацима!

![Грамофон](../../../5-Clustering/images/turntable.jpg)

> Фотографија од <a href="https://unsplash.com/@marcelalaskoski?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Marcela Laskoski</a> на <a href="https://unsplash.com/s/photos/nigerian-music?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
  
У овом низу лекција открићете нове начине за анализу података користећи технике кластерисања. Кластерисање је посебно корисно када вашем скупу података недостају ознаке. Ако има ознаке, онда би технике класификације, као оне које сте научили у претходним лекцијама, могле бити корисније. Али у случајевима када желите да групишете податке без ознака, кластерисање је одличан начин за откривање образаца.

> Постоје корисни алати са мало кода који вам могу помоћи да научите како да радите са моделима кластерисања. Пробајте [Azure ML за овај задатак](https://docs.microsoft.com/learn/modules/create-clustering-model-azure-machine-learning-designer/?WT.mc_id=academic-77952-leestott)

## Лекције

1. [Увод у кластерисање](1-Visualize/README.md)
2. [К-Меанс кластерисање](2-K-Means/README.md)

## Кредити

Ове лекције су написане са 🎶 од стране [Jen Looper](https://www.twitter.com/jenlooper) уз корисне рецензије од [Rishit Dagli](https://rishit_dagli) и [Muhammad Sakib Khan Inan](https://twitter.com/Sakibinan).

Скуп података [Нигеријске песме](https://www.kaggle.com/sootersaalu/nigerian-songs-spotify) је преузет са Kaggle-а као прикупљен са Spotify-а.

Корисни примери К-Меанс кластерисања који су помогли у креирању ове лекције укључују ову [анализу ириса](https://www.kaggle.com/bburns/iris-exploration-pca-k-means-and-gmm-clustering), овај [уводни нотебук](https://www.kaggle.com/prashant111/k-means-clustering-with-python), и овај [хипотетички пример НВО](https://www.kaggle.com/ankandash/pca-k-means-clustering-hierarchical-clustering).

---

**Одрицање од одговорности**:  
Овај документ је преведен коришћењем услуге за превођење помоћу вештачке интелигенције [Co-op Translator](https://github.com/Azure/co-op-translator). Иако се трудимо да обезбедимо тачност, молимо вас да имате у виду да аутоматски преводи могу садржати грешке или нетачности. Оригинални документ на његовом изворном језику треба сматрати ауторитативним извором. За критичне информације препоручује се професионални превод од стране људи. Не преузимамо одговорност за било каква погрешна тумачења или неспоразуме који могу настати услед коришћења овог превода.