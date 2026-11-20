# Modelos de clustering para aprendizagem automática

Clustering é uma tarefa de aprendizagem automática onde se procura encontrar objetos que se assemelham entre si e agrupá-los em grupos chamados clusters. O que diferencia o clustering de outros métodos na aprendizagem automática é que as coisas acontecem automaticamente, na verdade, é justo dizer que é o oposto da aprendizagem supervisionada.

## Tema regional: modelos de clustering para o gosto musical do público nigeriano 🎧

O público diverso da Nigéria tem gostos musicais diversos. Usando dados recolhidos do Spotify (inspirado por [este artigo](https://towardsdatascience.com/country-wise-visual-analysis-of-music-taste-using-spotify-api-seaborn-in-python-77f5b749b421), vejamos alguma música popular na Nigéria. Este conjunto de dados inclui informações sobre a pontuação de 'danceabilidade', 'acousticness', volume, 'speechiness', popularidade e energia de várias músicas. Será interessante descobrir padrões nestes dados!

![Uma gira-discos](../../../translated_images/pt-PT/turntable.f2b86b13c53302dc.webp)

> Foto de <a href="https://unsplash.com/@marcelalaskoski?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Marcela Laskoski</a> em <a href="https://unsplash.com/s/photos/nigerian-music?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
  
Nesta série de lições, vai descobrir novas formas de analisar dados usando técnicas de clustering. O clustering é particularmente útil quando o seu conjunto de dados não tem rótulos. Se os tiver, técnicas de classificação como aquelas que aprendeu em lições anteriores podem ser mais úteis. Mas nos casos em que procura agrupar dados não rotulados, o clustering é uma ótima forma de descobrir padrões.

> Existem ferramentas low-code úteis que podem ajudar a aprender a trabalhar com modelos de clustering. Experimente [Azure ML para esta tarefa](https://docs.microsoft.com/learn/modules/create-clustering-model-azure-machine-learning-designer/?WT.mc_id=academic-77952-leestott)

## Lições

1. [Introdução ao clustering](1-Visualize/README.md)
2. [Clustering K-Means](2-K-Means/README.md)

## Créditos

Estas lições foram escritas com 🎶 por [Jen Looper](https://www.twitter.com/jenlooper) com revisões úteis de [Rishit Dagli](https://rishit_dagli/) e [Muhammad Sakib Khan Inan](https://twitter.com/Sakibinan).

O conjunto de dados [Nigerian Songs](https://www.kaggle.com/sootersaalu/nigerian-songs-spotify) foi obtido na Kaggle, extraído do Spotify.

Exemplos úteis de K-Means que ajudaram na criação desta lição incluem esta [exploração do iris](https://www.kaggle.com/bburns/iris-exploration-pca-k-means-and-gmm-clustering), este [notebook introdutório](https://www.kaggle.com/prashant111/k-means-clustering-with-python), e este [exemplo hipotético de ONG](https://www.kaggle.com/ankandash/pca-k-means-clustering-hierarchical-clustering).

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Aviso Legal**:
Este documento foi traduzido utilizando o serviço de tradução automática [Co-op Translator](https://github.com/Azure/co-op-translator). Embora nos esforcemos pela precisão, esteja ciente de que traduções automáticas podem conter erros ou imprecisões. O documento original na sua língua nativa deve ser considerado a fonte autorizada. Para informações críticas, recomenda-se tradução profissional humana. Não nos responsabilizamos por quaisquer mal-entendidos ou interpretações incorretas resultantes da utilização desta tradução.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->