<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "b28a3a4911584062772c537b653ebbc7",
  "translation_date": "2025-08-29T20:52:39+00:00",
  "source_file": "5-Clustering/README.md",
  "language_code": "br"
}
-->
# Modelos de clustering para aprendizado de máquina

Clustering é uma tarefa de aprendizado de máquina que busca encontrar objetos que se assemelham entre si e agrupá-los em grupos chamados clusters. O que diferencia o clustering de outras abordagens no aprendizado de máquina é que tudo acontece automaticamente. Na verdade, é justo dizer que é o oposto do aprendizado supervisionado.

## Tópico regional: modelos de clustering para o gosto musical do público nigeriano 🎧

O público diversificado da Nigéria tem gostos musicais igualmente variados. Usando dados extraídos do Spotify (inspirado por [este artigo](https://towardsdatascience.com/country-wise-visual-analysis-of-music-taste-using-spotify-api-seaborn-in-python-77f5b749b421)), vamos analisar algumas músicas populares na Nigéria. Este conjunto de dados inclui informações sobre a pontuação de 'dançabilidade', 'acousticness', volume, 'speechiness', popularidade e energia de várias músicas. Será interessante descobrir padrões nesses dados!

![Um toca-discos](../../../translated_images/turntable.f2b86b13c53302dc106aa741de9dc96ac372864cf458dd6f879119857aab01da.br.jpg)

> Foto por <a href="https://unsplash.com/@marcelalaskoski?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Marcela Laskoski</a> no <a href="https://unsplash.com/s/photos/nigerian-music?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
  
Nesta série de lições, você descobrirá novas maneiras de analisar dados usando técnicas de clustering. O clustering é particularmente útil quando seu conjunto de dados não possui rótulos. Se ele tiver rótulos, então técnicas de classificação, como as que você aprendeu em lições anteriores, podem ser mais úteis. Mas, em casos onde você deseja agrupar dados não rotulados, o clustering é uma ótima maneira de descobrir padrões.

> Existem ferramentas de baixo código úteis que podem ajudá-lo a aprender a trabalhar com modelos de clustering. Experimente [Azure ML para esta tarefa](https://docs.microsoft.com/learn/modules/create-clustering-model-azure-machine-learning-designer/?WT.mc_id=academic-77952-leestott)

## Lições

1. [Introdução ao clustering](1-Visualize/README.md)
2. [Clustering com K-Means](2-K-Means/README.md)

## Créditos

Estas lições foram escritas com 🎶 por [Jen Looper](https://www.twitter.com/jenlooper) com revisões úteis de [Rishit Dagli](https://rishit_dagli) e [Muhammad Sakib Khan Inan](https://twitter.com/Sakibinan).

O conjunto de dados [Nigerian Songs](https://www.kaggle.com/sootersaalu/nigerian-songs-spotify) foi obtido no Kaggle, extraído do Spotify.

Exemplos úteis de K-Means que ajudaram na criação desta lição incluem esta [exploração de íris](https://www.kaggle.com/bburns/iris-exploration-pca-k-means-and-gmm-clustering), este [notebook introdutório](https://www.kaggle.com/prashant111/k-means-clustering-with-python) e este [exemplo hipotético de ONG](https://www.kaggle.com/ankandash/pca-k-means-clustering-hierarchical-clustering).

---

**Aviso Legal**:  
Este documento foi traduzido utilizando o serviço de tradução por IA [Co-op Translator](https://github.com/Azure/co-op-translator). Embora nos esforcemos para garantir a precisão, esteja ciente de que traduções automatizadas podem conter erros ou imprecisões. O documento original em seu idioma nativo deve ser considerado a fonte autoritativa. Para informações críticas, recomenda-se a tradução profissional realizada por humanos. Não nos responsabilizamos por quaisquer mal-entendidos ou interpretações equivocadas decorrentes do uso desta tradução.