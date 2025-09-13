<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "9d91f3af3758fdd4569fb410575995ef",
  "translation_date": "2025-09-05T08:44:28+00:00",
  "source_file": "1-Introduction/4-techniques-of-ML/README.md",
  "language_code": "pt"
}
-->
# Técnicas de Aprendizagem Automática

O processo de construir, utilizar e manter modelos de aprendizagem automática e os dados que eles utilizam é muito diferente de muitos outros fluxos de trabalho de desenvolvimento. Nesta lição, vamos desmistificar o processo e delinear as principais técnicas que precisa conhecer. Você irá:

- Compreender os processos que sustentam a aprendizagem automática a um nível elevado.
- Explorar conceitos básicos como 'modelos', 'previsões' e 'dados de treino'.

## [Questionário pré-aula](https://ff-quizzes.netlify.app/en/ml/)

[![ML para iniciantes - Técnicas de Aprendizagem Automática](https://img.youtube.com/vi/4NGM0U2ZSHU/0.jpg)](https://youtu.be/4NGM0U2ZSHU "ML para iniciantes - Técnicas de Aprendizagem Automática")

> 🎥 Clique na imagem acima para assistir a um vídeo curto sobre esta lição.

## Introdução

De forma geral, o processo de criação de processos de aprendizagem automática (ML) é composto por várias etapas:

1. **Definir a pergunta**. A maioria dos processos de ML começa com uma pergunta que não pode ser respondida por um programa condicional simples ou um motor baseado em regras. Estas perguntas geralmente giram em torno de previsões baseadas numa coleção de dados.
2. **Recolher e preparar os dados**. Para responder à sua pergunta, precisa de dados. A qualidade e, por vezes, a quantidade dos seus dados determinarão o quão bem pode responder à pergunta inicial. Visualizar os dados é um aspeto importante desta fase. Esta fase também inclui dividir os dados em grupos de treino e teste para construir um modelo.
3. **Escolher um método de treino**. Dependendo da sua pergunta e da natureza dos seus dados, precisa de escolher como deseja treinar um modelo para refletir melhor os seus dados e fazer previsões precisas. Esta é a parte do processo de ML que requer conhecimentos específicos e, muitas vezes, uma quantidade considerável de experimentação.
4. **Treinar o modelo**. Usando os seus dados de treino, utilizará vários algoritmos para treinar um modelo que reconheça padrões nos dados. O modelo pode usar pesos internos que podem ser ajustados para privilegiar certas partes dos dados em detrimento de outras, a fim de construir um modelo melhor.
5. **Avaliar o modelo**. Utiliza dados nunca antes vistos (os seus dados de teste) do conjunto recolhido para verificar o desempenho do modelo.
6. **Ajustar parâmetros**. Com base no desempenho do modelo, pode refazer o processo utilizando diferentes parâmetros ou variáveis que controlam o comportamento dos algoritmos usados para treinar o modelo.
7. **Prever**. Use novos dados de entrada para testar a precisão do modelo.

## Que pergunta fazer

Os computadores são particularmente habilidosos em descobrir padrões ocultos nos dados. Esta capacidade é muito útil para investigadores que têm perguntas sobre um determinado domínio que não podem ser facilmente respondidas criando um motor de regras condicionais. Dada uma tarefa atuarial, por exemplo, um cientista de dados pode ser capaz de construir regras manuais sobre a mortalidade de fumadores versus não fumadores.

Quando muitas outras variáveis são introduzidas na equação, no entanto, um modelo de ML pode revelar-se mais eficiente para prever taxas de mortalidade futuras com base no histórico de saúde passado. Um exemplo mais animador pode ser fazer previsões meteorológicas para o mês de abril numa determinada localização com base em dados que incluem latitude, longitude, alterações climáticas, proximidade ao oceano, padrões da corrente de jato, entre outros.

✅ Este [conjunto de slides](https://www2.cisl.ucar.edu/sites/default/files/2021-10/0900%20June%2024%20Haupt_0.pdf) sobre modelos meteorológicos oferece uma perspetiva histórica sobre o uso de ML na análise do clima.  

## Tarefas pré-construção

Antes de começar a construir o seu modelo, há várias tarefas que precisa de completar. Para testar a sua pergunta e formar uma hipótese com base nas previsões de um modelo, precisa de identificar e configurar vários elementos.

### Dados

Para responder à sua pergunta com algum grau de certeza, precisa de uma boa quantidade de dados do tipo certo. Há duas coisas que precisa de fazer neste momento:

- **Recolher dados**. Tendo em mente a lição anterior sobre justiça na análise de dados, recolha os seus dados com cuidado. Esteja atento às fontes desses dados, a quaisquer preconceitos inerentes que possam ter e documente a sua origem.
- **Preparar dados**. Há vários passos no processo de preparação de dados. Pode ser necessário reunir dados e normalizá-los se vierem de fontes diversas. Pode melhorar a qualidade e a quantidade dos dados através de vários métodos, como converter strings em números (como fazemos em [Clustering](../../5-Clustering/1-Visualize/README.md)). Também pode gerar novos dados com base nos originais (como fazemos em [Classificação](../../4-Classification/1-Introduction/README.md)). Pode limpar e editar os dados (como faremos antes da lição sobre [Aplicações Web](../../3-Web-App/README.md)). Por fim, pode ser necessário randomizar e embaralhar os dados, dependendo das suas técnicas de treino.

✅ Após recolher e processar os seus dados, reserve um momento para verificar se a sua estrutura permitirá abordar a pergunta pretendida. Pode ser que os dados não funcionem bem na sua tarefa, como descobrimos nas nossas lições de [Clustering](../../5-Clustering/1-Visualize/README.md)!

### Features e Target

Uma [feature](https://www.datasciencecentral.com/profiles/blogs/an-introduction-to-variable-and-feature-selection) é uma propriedade mensurável dos seus dados. Em muitos conjuntos de dados, é expressa como um cabeçalho de coluna, como 'data', 'tamanho' ou 'cor'. A sua variável de feature, geralmente representada como `X` no código, representa a variável de entrada que será usada para treinar o modelo.

Um target é aquilo que está a tentar prever. O target, geralmente representado como `y` no código, representa a resposta à pergunta que está a tentar fazer aos seus dados: em dezembro, qual será a **cor** das abóboras mais baratas? Em São Francisco, que bairros terão o melhor **preço** imobiliário? Às vezes, o target também é referido como atributo de rótulo.

### Selecionar a sua variável de feature

🎓 **Seleção de Features e Extração de Features** Como saber qual variável escolher ao construir um modelo? Provavelmente passará por um processo de seleção de features ou extração de features para escolher as variáveis certas para o modelo mais eficiente. No entanto, não são a mesma coisa: "A extração de features cria novas features a partir de funções das features originais, enquanto a seleção de features retorna um subconjunto das features." ([fonte](https://wikipedia.org/wiki/Feature_selection))

### Visualizar os seus dados

Um aspeto importante do conjunto de ferramentas de um cientista de dados é a capacidade de visualizar dados usando várias bibliotecas excelentes, como Seaborn ou MatPlotLib. Representar os seus dados visualmente pode permitir-lhe descobrir correlações ocultas que pode aproveitar. As suas visualizações também podem ajudá-lo a identificar preconceitos ou dados desequilibrados (como descobrimos em [Classificação](../../4-Classification/2-Classifiers-1/README.md)).

### Dividir o seu conjunto de dados

Antes de treinar, precisa de dividir o seu conjunto de dados em duas ou mais partes de tamanhos desiguais que ainda representem bem os dados.

- **Treino**. Esta parte do conjunto de dados é ajustada ao seu modelo para treiná-lo. Este conjunto constitui a maior parte do conjunto de dados original.
- **Teste**. Um conjunto de teste é um grupo independente de dados, muitas vezes retirado dos dados originais, que utiliza para confirmar o desempenho do modelo construído.
- **Validação**. Um conjunto de validação é um grupo independente menor de exemplos que utiliza para ajustar os hiperparâmetros ou a arquitetura do modelo, a fim de melhorá-lo. Dependendo do tamanho dos seus dados e da pergunta que está a fazer, pode não ser necessário construir este terceiro conjunto (como notamos em [Previsão de Séries Temporais](../../7-TimeSeries/1-Introduction/README.md)).

## Construir um modelo

Usando os seus dados de treino, o seu objetivo é construir um modelo, ou uma representação estatística dos seus dados, utilizando vários algoritmos para **treiná-lo**. Treinar um modelo expõe-no aos dados e permite-lhe fazer suposições sobre padrões percebidos que descobre, valida e aceita ou rejeita.

### Decidir sobre um método de treino

Dependendo da sua pergunta e da natureza dos seus dados, escolherá um método para treiná-lo. Explorando a [documentação do Scikit-learn](https://scikit-learn.org/stable/user_guide.html) - que usamos neste curso - pode descobrir várias formas de treinar um modelo. Dependendo da sua experiência, pode ter de experimentar vários métodos diferentes para construir o melhor modelo. É provável que passe por um processo em que os cientistas de dados avaliam o desempenho de um modelo alimentando-o com dados não vistos, verificando a precisão, preconceitos e outros problemas que degradam a qualidade, e selecionando o método de treino mais apropriado para a tarefa em questão.

### Treinar um modelo

Com os seus dados de treino, está pronto para 'ajustá-los' para criar um modelo. Notará que em muitas bibliotecas de ML encontrará o código 'model.fit' - é neste momento que envia a sua variável de feature como um array de valores (geralmente 'X') e uma variável target (geralmente 'y').

### Avaliar o modelo

Uma vez concluído o processo de treino (pode levar muitas iterações, ou 'épocas', para treinar um modelo grande), poderá avaliar a qualidade do modelo utilizando dados de teste para medir o seu desempenho. Estes dados são um subconjunto dos dados originais que o modelo ainda não analisou. Pode imprimir uma tabela de métricas sobre a qualidade do modelo.

🎓 **Ajuste do modelo**

No contexto da aprendizagem automática, o ajuste do modelo refere-se à precisão da função subjacente do modelo enquanto tenta analisar dados com os quais não está familiarizado.

🎓 **Subajuste** e **sobreajuste** são problemas comuns que degradam a qualidade do modelo, pois o modelo ajusta-se ou não suficientemente bem ou bem demais. Isso faz com que o modelo faça previsões muito alinhadas ou pouco alinhadas com os seus dados de treino. Um modelo sobreajustado prevê os dados de treino muito bem porque aprendeu os detalhes e o ruído dos dados em excesso. Um modelo subajustado não é preciso, pois não consegue analisar com precisão nem os dados de treino nem os dados que ainda não 'viu'.

![modelo sobreajustado](../../../../1-Introduction/4-techniques-of-ML/images/overfitting.png)
> Infográfico por [Jen Looper](https://twitter.com/jenlooper)

## Ajuste de parâmetros

Depois de concluir o treino inicial, observe a qualidade do modelo e considere melhorá-lo ajustando os seus 'hiperparâmetros'. Leia mais sobre o processo [na documentação](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters?WT.mc_id=academic-77952-leestott).

## Previsão

Este é o momento em que pode usar dados completamente novos para testar a precisão do seu modelo. Num cenário de ML 'aplicado', onde está a construir ativos web para usar o modelo em produção, este processo pode envolver a recolha de input do utilizador (um clique num botão, por exemplo) para definir uma variável e enviá-la ao modelo para inferência ou avaliação.

Nestes módulos, descobrirá como usar estes passos para preparar, construir, testar, avaliar e prever - todos os gestos de um cientista de dados e mais, à medida que progride na sua jornada para se tornar um engenheiro de ML 'full stack'.

---

## 🚀Desafio

Desenhe um fluxograma refletindo os passos de um praticante de ML. Onde se encontra neste momento no processo? Onde prevê que encontrará dificuldades? O que lhe parece fácil?

## [Questionário pós-aula](https://ff-quizzes.netlify.app/en/ml/)

## Revisão e Estudo Individual

Pesquise online entrevistas com cientistas de dados que discutam o seu trabalho diário. Aqui está [uma](https://www.youtube.com/watch?v=Z3IjgbbCEfs).

## Tarefa

[Entrevistar um cientista de dados](assignment.md)

---

**Aviso Legal**:  
Este documento foi traduzido utilizando o serviço de tradução por IA [Co-op Translator](https://github.com/Azure/co-op-translator). Embora nos esforcemos para garantir a precisão, esteja ciente de que traduções automáticas podem conter erros ou imprecisões. O documento original na sua língua nativa deve ser considerado a fonte oficial. Para informações críticas, recomenda-se uma tradução profissional realizada por humanos. Não nos responsabilizamos por quaisquer mal-entendidos ou interpretações incorretas resultantes do uso desta tradução.