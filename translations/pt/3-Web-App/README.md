<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "9836ff53cfef716ddfd70e06c5f43436",
  "translation_date": "2025-09-03T17:53:31+00:00",
  "source_file": "3-Web-App/README.md",
  "language_code": "pt"
}
-->
# Crie uma aplicação web para usar o seu modelo de ML

Nesta secção do currículo, será introduzido a um tópico aplicado de ML: como guardar o seu modelo Scikit-learn como um ficheiro que pode ser utilizado para fazer previsões dentro de uma aplicação web. Depois de guardar o modelo, aprenderá como utilizá-lo numa aplicação web construída em Flask. Primeiro, irá criar um modelo utilizando alguns dados relacionados com avistamentos de OVNIs! Em seguida, irá construir uma aplicação web que permitirá introduzir um número de segundos juntamente com valores de latitude e longitude para prever qual país relatou ter visto um OVNI.

![Estacionamento de OVNIs](../../../translated_images/ufo.9e787f5161da9d4d1dafc537e1da09be8210f2ee996cb638aa5cee1d92867a04.pt.jpg)

Foto por <a href="https://unsplash.com/@mdherren?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Michael Herren</a> em <a href="https://unsplash.com/s/photos/ufo?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>

## Lições

1. [Crie uma Aplicação Web](1-Web-App/README.md)

## Créditos

"Crie uma Aplicação Web" foi escrito com ♥️ por [Jen Looper](https://twitter.com/jenlooper).

♥️ Os questionários foram escritos por Rohan Raj.

O conjunto de dados foi obtido de [Kaggle](https://www.kaggle.com/NUFORC/ufo-sightings).

A arquitetura da aplicação web foi sugerida em parte por [este artigo](https://towardsdatascience.com/how-to-easily-deploy-machine-learning-models-using-flask-b95af8fe34d4) e [este repositório](https://github.com/abhinavsagar/machine-learning-deployment) de Abhinav Sagar.

---

**Aviso Legal**:  
Este documento foi traduzido utilizando o serviço de tradução por IA [Co-op Translator](https://github.com/Azure/co-op-translator). Embora nos esforcemos para garantir a precisão, esteja ciente de que traduções automáticas podem conter erros ou imprecisões. O documento original no seu idioma nativo deve ser considerado a fonte oficial. Para informações críticas, recomenda-se uma tradução profissional realizada por humanos. Não nos responsabilizamos por quaisquer mal-entendidos ou interpretações incorretas decorrentes do uso desta tradução.