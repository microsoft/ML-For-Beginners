<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "b28a3a4911584062772c537b653ebbc7",
  "translation_date": "2025-09-03T17:02:00+00:00",
  "source_file": "5-Clustering/README.md",
  "language_code": "pt"
}
-->
# Modelos de clustering para aprendizagem automática

Clustering é uma tarefa de aprendizagem automática que procura encontrar objetos semelhantes entre si e agrupá-los em grupos chamados clusters. O que diferencia o clustering de outras abordagens na aprendizagem automática é que tudo acontece de forma automática; na verdade, é justo dizer que é o oposto da aprendizagem supervisionada.

## Tópico regional: modelos de clustering para o gosto musical do público nigeriano 🎧

O público diversificado da Nigéria tem gostos musicais igualmente variados. Usando dados extraídos do Spotify (inspirado por [este artigo](https://towardsdatascience.com/country-wise-visual-analysis-of-music-taste-using-spotify-api-seaborn-in-python-77f5b749b421)), vamos analisar algumas músicas populares na Nigéria. Este conjunto de dados inclui informações sobre o 'danceability', 'acousticness', volume, 'speechiness', popularidade e energia de várias músicas. Será interessante descobrir padrões nesses dados!

![Um gira-discos](../../../translated_images/turntable.f2b86b13c53302dc106aa741de9dc96ac372864cf458dd6f879119857aab01da.pt.jpg)

> Foto de <a href="https://unsplash.com/@marcelalaskoski?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Marcela Laskoski</a> no <a href="https://unsplash.com/s/photos/nigerian-music?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
  
Nesta série de lições, vais descobrir novas formas de analisar dados utilizando técnicas de clustering. O clustering é particularmente útil quando o teu conjunto de dados não tem etiquetas. Se tiver etiquetas, então técnicas de classificação, como as que aprendeste em lições anteriores, podem ser mais úteis. Mas, em casos onde procuras agrupar dados não etiquetados, o clustering é uma ótima forma de descobrir padrões.

> Existem ferramentas úteis de baixo código que podem ajudar-te a aprender a trabalhar com modelos de clustering. Experimenta [Azure ML para esta tarefa](https://docs.microsoft.com/learn/modules/create-clustering-model-azure-machine-learning-designer/?WT.mc_id=academic-77952-leestott)

## Lições

1. [Introdução ao clustering](1-Visualize/README.md)
2. [Clustering com K-Means](2-K-Means/README.md)

## Créditos

Estas lições foram escritas com 🎶 por [Jen Looper](https://www.twitter.com/jenlooper) com revisões úteis de [Rishit Dagli](https://rishit_dagli) e [Muhammad Sakib Khan Inan](https://twitter.com/Sakibinan).

O conjunto de dados [Nigerian Songs](https://www.kaggle.com/sootersaalu/nigerian-songs-spotify) foi obtido no Kaggle, extraído do Spotify.

Exemplos úteis de K-Means que ajudaram na criação desta lição incluem esta [exploração de íris](https://www.kaggle.com/bburns/iris-exploration-pca-k-means-and-gmm-clustering), este [notebook introdutório](https://www.kaggle.com/prashant111/k-means-clustering-with-python) e este [exemplo hipotético de ONG](https://www.kaggle.com/ankandash/pca-k-means-clustering-hierarchical-clustering).

---

**Aviso Legal**:  
Este documento foi traduzido utilizando o serviço de tradução por IA [Co-op Translator](https://github.com/Azure/co-op-translator). Embora nos esforcemos para garantir a precisão, é importante notar que traduções automáticas podem conter erros ou imprecisões. O documento original na sua língua nativa deve ser considerado a fonte autoritária. Para informações críticas, recomenda-se a tradução profissional realizada por humanos. Não nos responsabilizamos por quaisquer mal-entendidos ou interpretações incorretas decorrentes da utilização desta tradução.