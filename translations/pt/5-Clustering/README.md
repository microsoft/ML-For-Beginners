# Modelos de agrupamento para aprendizado de máquina

O agrupamento é uma tarefa de aprendizado de máquina onde se busca encontrar objetos que se assemelham entre si e agrupá-los em grupos chamados clusters. O que diferencia o agrupamento de outras abordagens em aprendizado de máquina é que tudo acontece automaticamente; na verdade, é justo dizer que é o oposto do aprendizado supervisionado.

## Tópico regional: modelos de agrupamento para o gosto musical do público nigeriano 🎧

O público diversificado da Nigéria tem gostos musicais variados. Usando dados extraídos do Spotify (inspirado por [este artigo](https://towardsdatascience.com/country-wise-visual-analysis-of-music-taste-using-spotify-api-seaborn-in-python-77f5b749b421), vamos analisar algumas músicas populares na Nigéria. Este conjunto de dados inclui informações sobre a pontuação de 'dançabilidade' de várias músicas, 'acústica', volume, 'fala', popularidade e energia. Será interessante descobrir padrões nesses dados!

![Uma mesa de som](../../../translated_images/turntable.f2b86b13c53302dc106aa741de9dc96ac372864cf458dd6f879119857aab01da.pt.jpg)

> Foto de <a href="https://unsplash.com/@marcelalaskoski?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Marcela Laskoski</a> em <a href="https://unsplash.com/s/photos/nigerian-music?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
  
Nesta série de lições, você descobrirá novas maneiras de analisar dados usando técnicas de agrupamento. O agrupamento é particularmente útil quando seu conjunto de dados não possui rótulos. Se ele tiver rótulos, então técnicas de classificação, como as que você aprendeu em lições anteriores, podem ser mais úteis. Mas em casos onde você está buscando agrupar dados não rotulados, o agrupamento é uma ótima maneira de descobrir padrões.

> Existem ferramentas de baixo código úteis que podem ajudá-lo a aprender a trabalhar com modelos de agrupamento. Experimente [Azure ML para esta tarefa](https://docs.microsoft.com/learn/modules/create-clustering-model-azure-machine-learning-designer/?WT.mc_id=academic-77952-leestott)

## Lições

1. [Introdução ao agrupamento](1-Visualize/README.md)
2. [Agrupamento K-Means](2-K-Means/README.md)

## Créditos

Estas lições foram escritas com 🎶 por [Jen Looper](https://www.twitter.com/jenlooper) com revisões úteis de [Rishit Dagli](https://rishit_dagli) e [Muhammad Sakib Khan Inan](https://twitter.com/Sakibinan).

O conjunto de dados [Músicas Nigerianas](https://www.kaggle.com/sootersaalu/nigerian-songs-spotify) foi obtido do Kaggle, conforme extraído do Spotify.

Exemplos úteis de K-Means que auxiliaram na criação desta lição incluem esta [exploração de íris](https://www.kaggle.com/bburns/iris-exploration-pca-k-means-and-gmm-clustering), este [caderno introdutório](https://www.kaggle.com/prashant111/k-means-clustering-with-python) e este [exemplo hipotético de ONG](https://www.kaggle.com/ankandash/pca-k-means-clustering-hierarchical-clustering).

**Isenção de responsabilidade**:  
Este documento foi traduzido utilizando serviços de tradução automática baseados em IA. Embora nos esforcemos pela precisão, esteja ciente de que traduções automatizadas podem conter erros ou imprecisões. O documento original em seu idioma nativo deve ser considerado a fonte autorizada. Para informações críticas, recomenda-se a tradução profissional realizada por humanos. Não nos responsabilizamos por quaisquer mal-entendidos ou interpretações errôneas resultantes do uso desta tradução.