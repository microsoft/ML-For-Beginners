<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "dc4575225da159f2b06706e103ddba2a",
  "translation_date": "2025-09-03T17:42:51+00:00",
  "source_file": "1-Introduction/4-techniques-of-ML/README.md",
  "language_code": "pt"
}
-->
# Técnicas de Aprendizagem Automática

O processo de construir, usar e manter modelos de aprendizagem automática e os dados que eles utilizam é muito diferente de muitos outros fluxos de trabalho de desenvolvimento. Nesta lição, vamos desmistificar o processo e delinear as principais técnicas que você precisa conhecer. Você irá:

- Compreender os processos que sustentam a aprendizagem automática em um nível geral.
- Explorar conceitos básicos como 'modelos', 'previsões' e 'dados de treino'.

## [Questionário pré-aula](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/7/)

[![ML para iniciantes - Técnicas de Aprendizagem Automática](https://img.youtube.com/vi/4NGM0U2ZSHU/0.jpg)](https://youtu.be/4NGM0U2ZSHU "ML para iniciantes - Técnicas de Aprendizagem Automática")

> 🎥 Clique na imagem acima para assistir a um vídeo curto sobre esta lição.

## Introdução

De forma geral, o processo de criar fluxos de aprendizagem automática (ML) é composto por várias etapas:

1. **Definir a pergunta**. A maioria dos processos de ML começa com uma pergunta que não pode ser respondida por um programa condicional simples ou um motor baseado em regras. Essas perguntas geralmente giram em torno de previsões baseadas em um conjunto de dados.
2. **Coletar e preparar os dados**. Para responder à sua pergunta, você precisa de dados. A qualidade e, às vezes, a quantidade dos seus dados determinarão o quão bem você pode responder à pergunta inicial. Visualizar os dados é um aspecto importante desta fase. Esta etapa também inclui dividir os dados em grupos de treino e teste para construir um modelo.
3. **Escolher um método de treino**. Dependendo da sua pergunta e da natureza dos seus dados, você precisa escolher como deseja treinar um modelo para refletir melhor os dados e fazer previsões precisas. Esta é a parte do processo de ML que exige expertise específica e, frequentemente, uma quantidade considerável de experimentação.
4. **Treinar o modelo**. Usando os seus dados de treino, você utilizará vários algoritmos para treinar um modelo que reconheça padrões nos dados. O modelo pode usar pesos internos que podem ser ajustados para privilegiar certas partes dos dados em detrimento de outras, a fim de construir um modelo melhor.
5. **Avaliar o modelo**. Você usa dados nunca antes vistos (os seus dados de teste) do conjunto coletado para verificar o desempenho do modelo.
6. **Ajustar parâmetros**. Com base no desempenho do modelo, você pode refazer o processo usando diferentes parâmetros ou variáveis que controlam o comportamento dos algoritmos usados para treinar o modelo.
7. **Prever**. Use novas entradas para testar a precisão do modelo.

## Qual pergunta fazer

Os computadores são particularmente habilidosos em descobrir padrões ocultos nos dados. Essa utilidade é muito útil para pesquisadores que têm perguntas sobre um determinado domínio que não podem ser facilmente respondidas criando um motor baseado em regras condicionais. Dado um trabalho atuarial, por exemplo, um cientista de dados pode ser capaz de construir regras personalizadas sobre a mortalidade de fumadores versus não fumadores.

Quando muitas outras variáveis são introduzidas na equação, no entanto, um modelo de ML pode ser mais eficiente para prever taxas de mortalidade futuras com base no histórico de saúde anterior. Um exemplo mais animador pode ser fazer previsões meteorológicas para o mês de abril em um determinado local com base em dados que incluem latitude, longitude, mudanças climáticas, proximidade ao oceano, padrões de correntes de jato e mais.

✅ Este [slide deck](https://www2.cisl.ucar.edu/sites/default/files/2021-10/0900%20June%2024%20Haupt_0.pdf) sobre modelos meteorológicos oferece uma perspectiva histórica sobre o uso de ML na análise do clima.  

## Tarefas pré-construção

Antes de começar a construir o seu modelo, há várias tarefas que você precisa completar. Para testar a sua pergunta e formar uma hipótese com base nas previsões de um modelo, você precisa identificar e configurar vários elementos.

### Dados

Para responder à sua pergunta com algum grau de certeza, você precisa de uma boa quantidade de dados do tipo certo. Há duas coisas que você precisa fazer neste momento:

- **Coletar dados**. Lembrando a lição anterior sobre justiça na análise de dados, colete os seus dados com cuidado. Esteja atento às fontes desses dados, a quaisquer preconceitos inerentes que eles possam ter e documente a sua origem.
- **Preparar dados**. Há várias etapas no processo de preparação de dados. Você pode precisar reunir dados e normalizá-los se eles vierem de fontes diversas. Pode melhorar a qualidade e a quantidade dos dados através de vários métodos, como converter strings em números (como fazemos em [Clustering](../../5-Clustering/1-Visualize/README.md)). Também pode gerar novos dados com base nos originais (como fazemos em [Classificação](../../4-Classification/1-Introduction/README.md)). Pode limpar e editar os dados (como faremos antes da lição de [Aplicação Web](../../3-Web-App/README.md)). Finalmente, pode ser necessário randomizar e embaralhar os dados, dependendo das técnicas de treino.

✅ Após coletar e processar os seus dados, reserve um momento para verificar se o formato deles permitirá que você responda à pergunta pretendida. Pode ser que os dados não funcionem bem na sua tarefa, como descobrimos nas lições de [Clustering](../../5-Clustering/1-Visualize/README.md)!

### Features e Target

Uma [feature](https://www.datasciencecentral.com/profiles/blogs/an-introduction-to-variable-and-feature-selection) é uma propriedade mensurável dos seus dados. Em muitos conjuntos de dados, ela é expressa como um cabeçalho de coluna, como 'data', 'tamanho' ou 'cor'. A sua variável de feature, geralmente representada como `X` no código, representa a variável de entrada que será usada para treinar o modelo.

Um target é aquilo que você está tentando prever. O target, geralmente representado como `y` no código, representa a resposta à pergunta que você está tentando fazer aos seus dados: em dezembro, qual **cor** de abóbora será mais barata? Em São Francisco, quais bairros terão o melhor **preço** imobiliário? Às vezes, o target também é referido como atributo de rótulo.

### Selecionar a sua variável de feature

🎓 **Seleção de Features e Extração de Features** Como saber qual variável escolher ao construir um modelo? Provavelmente, você passará por um processo de seleção ou extração de features para escolher as variáveis certas para o modelo mais eficiente. No entanto, elas não são a mesma coisa: "A extração de features cria novas features a partir de funções das features originais, enquanto a seleção de features retorna um subconjunto das features." ([fonte](https://wikipedia.org/wiki/Feature_selection))

### Visualizar os seus dados

Um aspecto importante do kit de ferramentas do cientista de dados é o poder de visualizar dados usando várias bibliotecas excelentes, como Seaborn ou MatPlotLib. Representar os seus dados visualmente pode permitir que você descubra correlações ocultas que pode aproveitar. As suas visualizações também podem ajudar a identificar preconceitos ou dados desequilibrados (como descobrimos em [Classificação](../../4-Classification/2-Classifiers-1/README.md)).

### Dividir o seu conjunto de dados

Antes de treinar, você precisa dividir o seu conjunto de dados em duas ou mais partes de tamanhos desiguais que ainda representem bem os dados.

- **Treino**. Esta parte do conjunto de dados é ajustada ao seu modelo para treiná-lo. Este conjunto constitui a maior parte do conjunto de dados original.
- **Teste**. Um conjunto de teste é um grupo independente de dados, frequentemente retirado dos dados originais, que você usa para confirmar o desempenho do modelo construído.
- **Validação**. Um conjunto de validação é um grupo menor e independente de exemplos que você usa para ajustar os hiperparâmetros ou a arquitetura do modelo para melhorá-lo. Dependendo do tamanho dos seus dados e da pergunta que está fazendo, pode não ser necessário construir este terceiro conjunto (como observamos em [Previsão de Séries Temporais](../../7-TimeSeries/1-Introduction/README.md)).

## Construindo um modelo

Usando os seus dados de treino, o seu objetivo é construir um modelo, ou uma representação estatística dos seus dados, utilizando vários algoritmos para **treiná-lo**. Treinar um modelo expõe-o aos dados e permite que ele faça suposições sobre padrões percebidos que descobre, valida e aceita ou rejeita.

### Decidir sobre um método de treino

Dependendo da sua pergunta e da natureza dos seus dados, você escolherá um método para treiná-lo. Explorando a [documentação do Scikit-learn](https://scikit-learn.org/stable/user_guide.html) - que usamos neste curso - você pode descobrir várias maneiras de treinar um modelo. Dependendo da sua experiência, pode ser necessário tentar vários métodos diferentes para construir o melhor modelo. É provável que você passe por um processo em que cientistas de dados avaliam o desempenho de um modelo alimentando-o com dados não vistos, verificando a precisão, preconceitos e outros problemas que degradam a qualidade, e selecionando o método de treino mais apropriado para a tarefa em questão.

### Treinar um modelo

Com os seus dados de treino em mãos, você está pronto para 'ajustá-los' e criar um modelo. Você notará que em muitas bibliotecas de ML encontrará o código 'model.fit' - é neste momento que você envia a sua variável de feature como um array de valores (geralmente 'X') e uma variável de target (geralmente 'y').

### Avaliar o modelo

Uma vez concluído o processo de treino (pode levar muitas iterações, ou 'épocas', para treinar um modelo grande), você poderá avaliar a qualidade do modelo usando dados de teste para medir o seu desempenho. Esses dados são um subconjunto dos dados originais que o modelo ainda não analisou. Você pode imprimir uma tabela de métricas sobre a qualidade do modelo.

🎓 **Ajuste do modelo**

No contexto da aprendizagem automática, o ajuste do modelo refere-se à precisão da função subjacente do modelo ao tentar analisar dados com os quais não está familiarizado.

🎓 **Subajuste** e **sobreajuste** são problemas comuns que degradam a qualidade do modelo, pois ele se ajusta ou não suficientemente bem ou excessivamente bem. Isso faz com que o modelo faça previsões muito alinhadas ou pouco alinhadas com os seus dados de treino. Um modelo sobreajustado prevê os dados de treino muito bem porque aprendeu os detalhes e ruídos dos dados excessivamente bem. Um modelo subajustado não é preciso, pois não consegue analisar com precisão nem os seus dados de treino nem os dados que ainda não 'viu'.

![modelo sobreajustado](../../../../translated_images/overfitting.1c132d92bfd93cb63240baf63ebdf82c30e30a0a44e1ad49861b82ff600c2b5c.pt.png)
> Infográfico por [Jen Looper](https://twitter.com/jenlooper)

## Ajuste de parâmetros

Depois de concluir o treino inicial, observe a qualidade do modelo e considere melhorá-lo ajustando os seus 'hiperparâmetros'. Leia mais sobre o processo [na documentação](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters?WT.mc_id=academic-77952-leestott).

## Previsão

Este é o momento em que você pode usar dados completamente novos para testar a precisão do modelo. Em um cenário de ML 'aplicado', onde você está construindo ativos web para usar o modelo em produção, este processo pode envolver a coleta de entrada do usuário (um clique de botão, por exemplo) para definir uma variável e enviá-la ao modelo para inferência ou avaliação.

Nestes módulos, você descobrirá como usar estas etapas para preparar, construir, testar, avaliar e prever - todos os gestos de um cientista de dados e mais, à medida que avança na sua jornada para se tornar um engenheiro de ML 'full stack'.

---

## 🚀Desafio

Desenhe um fluxograma refletindo os passos de um praticante de ML. Onde você se vê agora no processo? Onde prevê que encontrará dificuldades? O que parece fácil para você?

## [Questionário pós-aula](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/8/)

## Revisão & Autoestudo

Pesquise online entrevistas com cientistas de dados que discutem o seu trabalho diário. Aqui está [uma](https://www.youtube.com/watch?v=Z3IjgbbCEfs).

## Tarefa

[Entrevistar um cientista de dados](assignment.md)

---

**Aviso Legal**:  
Este documento foi traduzido utilizando o serviço de tradução por IA [Co-op Translator](https://github.com/Azure/co-op-translator). Embora nos esforcemos para garantir a precisão, é importante notar que traduções automáticas podem conter erros ou imprecisões. O documento original na sua língua nativa deve ser considerado a fonte autoritária. Para informações críticas, recomenda-se a tradução profissional realizada por humanos. Não nos responsabilizamos por quaisquer mal-entendidos ou interpretações incorretas decorrentes da utilização desta tradução.