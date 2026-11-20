# Modelos de clustering para aprendizado de máquina

Clustering é uma tarefa de aprendizado de máquina que busca encontrar objetos que se parecem entre si e agrupá-los em grupos chamados clusters. O que diferencia o clustering de outras abordagens em aprendizado de máquina é que as coisas acontecem automaticamente, na verdade, é justo dizer que é o oposto do aprendizado supervisionado.

## Tópico regional: modelos de clustering para o gosto musical do público nigeriano 🎧

O público diversificado da Nigéria tem gostos musicais variados. Usando dados extraídos do Spotify (inspirados por [este artigo](https://towardsdatascience.com/country-wise-visual-analysis-of-music-taste-using-spotify-api-seaborn-in-python-77f5b749b421)), vamos analisar algumas músicas populares na Nigéria. Este conjunto de dados inclui informações sobre a pontuação de 'dançabilidade', 'acústica', volume, 'fala', popularidade e energia de várias músicas. Será interessante descobrir padrões nesses dados!

![Um toca-discos](../../../translated_images/pt-BR/turntable.f2b86b13c53302dc.webp)

> Foto por <a href="https://unsplash.com/@marcelalaskoski?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Marcela Laskoski</a> em <a href="https://unsplash.com/s/photos/nigerian-music?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
  
Nesta série de lições, você descobrirá novas formas de analisar dados usando técnicas de clustering. Clustering é particularmente útil quando o seu conjunto de dados não possui rótulos. Se ele tiver rótulos, então técnicas de classificação como as que você aprendeu nas lições anteriores podem ser mais úteis. Mas nos casos em que você quer agrupar dados não rotulados, o clustering é uma ótima maneira de descobrir padrões.

> Existem ferramentas low-code úteis que podem ajudar você a aprender a trabalhar com modelos de clustering. Experimente [Azure ML para esta tarefa](https://docs.microsoft.com/learn/modules/create-clustering-model-azure-machine-learning-designer/?WT.mc_id=academic-77952-leestott)

## Lições

1. [Introdução ao clustering](1-Visualize/README.md)
2. [Clustering K-Means](2-K-Means/README.md)

## Créditos

Estas lições foram escritas com 🎶 por [Jen Looper](https://www.twitter.com/jenlooper) com revisões úteis de [Rishit Dagli](https://rishit_dagli/) e [Muhammad Sakib Khan Inan](https://twitter.com/Sakibinan).

O conjunto de dados [Nigerian Songs](https://www.kaggle.com/sootersaalu/nigerian-songs-spotify) foi obtido do Kaggle, extraído do Spotify.

Exemplos úteis de K-Means que auxiliaram na criação desta lição incluem esta [exploração do índris](https://www.kaggle.com/bburns/iris-exploration-pca-k-means-and-gmm-clustering), este [notebook introdutório](https://www.kaggle.com/prashant111/k-means-clustering-with-python) e este [exemplo hipotético de ONG](https://www.kaggle.com/ankandash/pca-k-means-clustering-hierarchical-clustering).

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Aviso Legal**:
Este documento foi traduzido usando o serviço de tradução por IA [Co-op Translator](https://github.com/Azure/co-op-translator). Embora nos esforcemos pela precisão, por favor, esteja ciente de que traduções automatizadas podem conter erros ou imprecisões. O documento original em seu idioma nativo deve ser considerado a fonte autorizada. Para informações críticas, recomenda-se tradução profissional humana. Não nos responsabilizamos por quaisquer mal-entendidos ou interpretações incorretas decorrentes do uso desta tradução.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->