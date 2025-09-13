<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "69389392fa6346e0dfa30f664b7b6fec",
  "translation_date": "2025-09-05T08:44:57+00:00",
  "source_file": "1-Introduction/1-intro-to-ML/README.md",
  "language_code": "pt"
}
-->
# Introdução ao machine learning

## [Questionário pré-aula](https://ff-quizzes.netlify.app/en/ml/)

---

[![ML para iniciantes - Introdução ao Machine Learning para Iniciantes](https://img.youtube.com/vi/6mSx_KJxcHI/0.jpg)](https://youtu.be/6mSx_KJxcHI "ML para iniciantes - Introdução ao Machine Learning para Iniciantes")

> 🎥 Clique na imagem acima para assistir a um vídeo curto sobre esta lição.

Bem-vindo a este curso sobre machine learning clássico para iniciantes! Quer seja completamente novo neste tema ou um praticante experiente de ML que procura rever uma área, estamos felizes por tê-lo connosco! Queremos criar um ponto de partida amigável para o seu estudo de ML e ficaremos felizes em avaliar, responder e incorporar o seu [feedback](https://github.com/microsoft/ML-For-Beginners/discussions).

[![Introdução ao ML](https://img.youtube.com/vi/h0e2HAPTGF4/0.jpg)](https://youtu.be/h0e2HAPTGF4 "Introdução ao ML")

> 🎥 Clique na imagem acima para assistir a um vídeo: John Guttag do MIT apresenta o machine learning

---
## Começando com machine learning

Antes de começar com este currículo, é necessário configurar o seu computador e prepará-lo para executar notebooks localmente.

- **Configure o seu computador com estes vídeos**. Utilize os seguintes links para aprender [como instalar Python](https://youtu.be/CXZYvNRIAKM) no seu sistema e [configurar um editor de texto](https://youtu.be/EU8eayHWoZg) para desenvolvimento.
- **Aprenda Python**. Também é recomendado ter uma compreensão básica de [Python](https://docs.microsoft.com/learn/paths/python-language/?WT.mc_id=academic-77952-leestott), uma linguagem de programação útil para cientistas de dados que utilizamos neste curso.
- **Aprenda Node.js e JavaScript**. Utilizamos JavaScript algumas vezes neste curso ao criar aplicações web, por isso será necessário ter [node](https://nodejs.org) e [npm](https://www.npmjs.com/) instalados, bem como [Visual Studio Code](https://code.visualstudio.com/) disponível para desenvolvimento em Python e JavaScript.
- **Crie uma conta no GitHub**. Como nos encontrou aqui no [GitHub](https://github.com), talvez já tenha uma conta, mas, se não, crie uma e depois faça um fork deste currículo para usar por conta própria. (Sinta-se à vontade para nos dar uma estrela também 😊)
- **Explore o Scikit-learn**. Familiarize-se com [Scikit-learn](https://scikit-learn.org/stable/user_guide.html), um conjunto de bibliotecas de ML que referenciamos nestas lições.

---
## O que é machine learning?

O termo 'machine learning' é um dos mais populares e frequentemente utilizados atualmente. Existe uma possibilidade significativa de que já tenha ouvido este termo pelo menos uma vez, caso tenha algum tipo de familiaridade com tecnologia, independentemente da área em que trabalha. No entanto, a mecânica do machine learning é um mistério para a maioria das pessoas. Para um iniciante em machine learning, o tema pode, por vezes, parecer avassalador. Por isso, é importante entender o que realmente é machine learning e aprender sobre ele passo a passo, através de exemplos práticos.

---
## A curva de hype

![ml hype curve](../../../../1-Introduction/1-intro-to-ML/images/hype.png)

> O Google Trends mostra a recente 'curva de hype' do termo 'machine learning'

---
## Um universo misterioso

Vivemos num universo cheio de mistérios fascinantes. Grandes cientistas como Stephen Hawking, Albert Einstein e muitos outros dedicaram as suas vidas à busca de informações significativas que desvendassem os mistérios do mundo ao nosso redor. Esta é a condição humana de aprender: uma criança humana aprende coisas novas e descobre a estrutura do seu mundo ano após ano, à medida que cresce até à idade adulta.

---
## O cérebro da criança

O cérebro e os sentidos de uma criança percebem os factos do seu ambiente e, gradualmente, aprendem os padrões ocultos da vida que ajudam a criança a criar regras lógicas para identificar padrões aprendidos. O processo de aprendizagem do cérebro humano torna os humanos a criatura viva mais sofisticada deste mundo. Aprender continuamente, descobrindo padrões ocultos e depois inovando com base nesses padrões, permite-nos melhorar cada vez mais ao longo da nossa vida. Esta capacidade de aprendizagem e evolução está relacionada a um conceito chamado [plasticidade cerebral](https://www.simplypsychology.org/brain-plasticity.html). Superficialmente, podemos traçar algumas semelhanças motivacionais entre o processo de aprendizagem do cérebro humano e os conceitos de machine learning.

---
## O cérebro humano

O [cérebro humano](https://www.livescience.com/29365-human-brain.html) percebe coisas do mundo real, processa as informações percebidas, toma decisões racionais e realiza certas ações com base nas circunstâncias. Isto é o que chamamos de comportamento inteligente. Quando programamos uma réplica do processo de comportamento inteligente numa máquina, chamamos isso de inteligência artificial (IA).

---
## Alguns termos

Embora os termos possam ser confundidos, machine learning (ML) é um subconjunto importante da inteligência artificial. **ML está relacionado ao uso de algoritmos especializados para descobrir informações significativas e encontrar padrões ocultos a partir de dados percebidos, corroborando o processo de tomada de decisão racional**.

---
## IA, ML, Deep Learning

![AI, ML, deep learning, data science](../../../../1-Introduction/1-intro-to-ML/images/ai-ml-ds.png)

> Um diagrama mostrando as relações entre IA, ML, deep learning e ciência de dados. Infográfico por [Jen Looper](https://twitter.com/jenlooper) inspirado por [este gráfico](https://softwareengineering.stackexchange.com/questions/366996/distinction-between-ai-ml-neural-networks-deep-learning-and-data-mining)

---
## Conceitos a abordar

Neste currículo, vamos abordar apenas os conceitos principais de machine learning que um iniciante deve conhecer. Abordamos o que chamamos de 'machine learning clássico', utilizando principalmente o Scikit-learn, uma excelente biblioteca que muitos estudantes utilizam para aprender o básico. Para entender conceitos mais amplos de inteligência artificial ou deep learning, um conhecimento fundamental sólido de machine learning é indispensável, e é isso que queremos oferecer aqui.

---
## Neste curso, irá aprender:

- conceitos principais de machine learning
- a história do ML
- ML e equidade
- técnicas de regressão em ML
- técnicas de classificação em ML
- técnicas de clustering em ML
- técnicas de processamento de linguagem natural em ML
- técnicas de previsão de séries temporais em ML
- aprendizagem por reforço
- aplicações reais de ML

---
## O que não iremos abordar

- deep learning
- redes neurais
- IA

Para proporcionar uma melhor experiência de aprendizagem, evitaremos as complexidades das redes neurais, 'deep learning' - construção de modelos com várias camadas utilizando redes neurais - e IA, que discutiremos num currículo diferente. Também ofereceremos um futuro currículo de ciência de dados para focar nesse aspeto deste campo mais amplo.

---
## Por que estudar machine learning?

Machine learning, do ponto de vista de sistemas, é definido como a criação de sistemas automatizados que podem aprender padrões ocultos a partir de dados para ajudar na tomada de decisões inteligentes.

Esta motivação é vagamente inspirada por como o cérebro humano aprende certas coisas com base nos dados que percebe do mundo exterior.

✅ Pense por um momento por que uma empresa gostaria de tentar usar estratégias de machine learning em vez de criar um motor baseado em regras codificadas.

---
## Aplicações de machine learning

As aplicações de machine learning estão agora quase em todo lugar e são tão ubíquas quanto os dados que circulam pelas nossas sociedades, gerados pelos nossos smartphones, dispositivos conectados e outros sistemas. Considerando o imenso potencial dos algoritmos de machine learning de última geração, os investigadores têm explorado a sua capacidade para resolver problemas reais multidimensionais e multidisciplinares com resultados muito positivos.

---
## Exemplos de ML aplicado

**Pode usar machine learning de várias formas**:

- Para prever a probabilidade de uma doença com base no histórico médico ou relatórios de um paciente.
- Para utilizar dados meteorológicos e prever eventos climáticos.
- Para entender o sentimento de um texto.
- Para detetar notícias falsas e impedir a propagação de propaganda.

Finanças, economia, ciência da terra, exploração espacial, engenharia biomédica, ciência cognitiva e até áreas das humanidades têm adaptado o machine learning para resolver os problemas árduos e pesados em processamento de dados dos seus domínios.

---
## Conclusão

Machine learning automatiza o processo de descoberta de padrões ao encontrar insights significativos a partir de dados reais ou gerados. Provou ser altamente valioso em aplicações empresariais, de saúde e financeiras, entre outras.

Num futuro próximo, entender os fundamentos de machine learning será indispensável para pessoas de qualquer área devido à sua ampla adoção.

---
# 🚀 Desafio

Desenhe, em papel ou utilizando uma aplicação online como [Excalidraw](https://excalidraw.com/), a sua compreensão das diferenças entre IA, ML, deep learning e ciência de dados. Adicione algumas ideias sobre os problemas que cada uma destas técnicas é boa em resolver.

# [Questionário pós-aula](https://ff-quizzes.netlify.app/en/ml/)

---
# Revisão e Autoestudo

Para aprender mais sobre como pode trabalhar com algoritmos de ML na nuvem, siga este [Percurso de Aprendizagem](https://docs.microsoft.com/learn/paths/create-no-code-predictive-models-azure-machine-learning/?WT.mc_id=academic-77952-leestott).

Faça um [Percurso de Aprendizagem](https://docs.microsoft.com/learn/modules/introduction-to-machine-learning/?WT.mc_id=academic-77952-leestott) sobre os fundamentos de ML.

---
# Tarefa

[Prepare-se e comece](assignment.md)

---

**Aviso Legal**:  
Este documento foi traduzido utilizando o serviço de tradução por IA [Co-op Translator](https://github.com/Azure/co-op-translator). Embora nos esforcemos para garantir a precisão, é importante notar que traduções automáticas podem conter erros ou imprecisões. O documento original na sua língua nativa deve ser considerado a fonte autoritária. Para informações críticas, recomenda-se a tradução profissional realizada por humanos. Não nos responsabilizamos por quaisquer mal-entendidos ou interpretações incorretas decorrentes da utilização desta tradução.