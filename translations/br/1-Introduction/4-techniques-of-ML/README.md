<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "9d91f3af3758fdd4569fb410575995ef",
  "translation_date": "2025-09-04T21:35:13+00:00",
  "source_file": "1-Introduction/4-techniques-of-ML/README.md",
  "language_code": "br"
}
-->
# Técnicas de Aprendizado de Máquina

O processo de construir, usar e manter modelos de aprendizado de máquina e os dados que eles utilizam é muito diferente de muitos outros fluxos de trabalho de desenvolvimento. Nesta lição, vamos desmistificar o processo e delinear as principais técnicas que você precisa conhecer. Você irá:

- Compreender os processos que sustentam o aprendizado de máquina em um nível geral.
- Explorar conceitos básicos como 'modelos', 'previsões' e 'dados de treinamento'.

## [Quiz pré-aula](https://ff-quizzes.netlify.app/en/ml/)

[![ML para iniciantes - Técnicas de Aprendizado de Máquina](https://img.youtube.com/vi/4NGM0U2ZSHU/0.jpg)](https://youtu.be/4NGM0U2ZSHU "ML para iniciantes - Técnicas de Aprendizado de Máquina")

> 🎥 Clique na imagem acima para assistir a um vídeo curto sobre esta lição.

## Introdução

Em um nível geral, a prática de criar processos de aprendizado de máquina (ML) é composta por várias etapas:

1. **Definir a pergunta**. A maioria dos processos de ML começa com uma pergunta que não pode ser respondida por um programa condicional simples ou um mecanismo baseado em regras. Essas perguntas geralmente giram em torno de previsões baseadas em um conjunto de dados.
2. **Coletar e preparar os dados**. Para responder à sua pergunta, você precisa de dados. A qualidade e, às vezes, a quantidade dos seus dados determinarão o quão bem você pode responder à pergunta inicial. Visualizar os dados é um aspecto importante desta fase. Esta etapa também inclui dividir os dados em grupos de treinamento e teste para construir um modelo.
3. **Escolher um método de treinamento**. Dependendo da sua pergunta e da natureza dos seus dados, você precisa escolher como deseja treinar um modelo para refletir melhor seus dados e fazer previsões precisas. Esta é a parte do processo de ML que exige expertise específica e, frequentemente, uma quantidade considerável de experimentação.
4. **Treinar o modelo**. Usando seus dados de treinamento, você aplicará vários algoritmos para treinar um modelo que reconheça padrões nos dados. O modelo pode usar pesos internos que podem ser ajustados para privilegiar certas partes dos dados em detrimento de outras, a fim de construir um modelo melhor.
5. **Avaliar o modelo**. Você usa dados nunca antes vistos (seus dados de teste) do conjunto coletado para verificar o desempenho do modelo.
6. **Ajustar parâmetros**. Com base no desempenho do modelo, você pode refazer o processo usando diferentes parâmetros ou variáveis que controlam o comportamento dos algoritmos usados para treinar o modelo.
7. **Prever**. Use novas entradas para testar a precisão do modelo.

## Qual pergunta fazer

Os computadores são particularmente habilidosos em descobrir padrões ocultos nos dados. Essa utilidade é muito útil para pesquisadores que têm perguntas sobre um determinado domínio que não podem ser facilmente respondidas criando um mecanismo baseado em regras condicionais. Dado um trabalho atuarial, por exemplo, um cientista de dados pode construir regras personalizadas sobre a mortalidade de fumantes versus não fumantes.

Quando muitas outras variáveis são incluídas na equação, no entanto, um modelo de ML pode ser mais eficiente para prever taxas de mortalidade futuras com base no histórico de saúde anterior. Um exemplo mais animador pode ser fazer previsões meteorológicas para o mês de abril em um determinado local com base em dados que incluem latitude, longitude, mudanças climáticas, proximidade ao oceano, padrões de correntes de jato e mais.

✅ Este [slide deck](https://www2.cisl.ucar.edu/sites/default/files/2021-10/0900%20June%2024%20Haupt_0.pdf) sobre modelos meteorológicos oferece uma perspectiva histórica sobre o uso de ML na análise climática.  

## Tarefas pré-construção

Antes de começar a construir seu modelo, há várias tarefas que você precisa completar. Para testar sua pergunta e formar uma hipótese com base nas previsões de um modelo, você precisa identificar e configurar vários elementos.

### Dados

Para responder à sua pergunta com algum grau de certeza, você precisa de uma boa quantidade de dados do tipo certo. Há duas coisas que você precisa fazer neste momento:

- **Coletar dados**. Lembre-se da lição anterior sobre justiça na análise de dados e colete seus dados com cuidado. Esteja atento às fontes desses dados, quaisquer vieses inerentes que possam ter e documente sua origem.
- **Preparar dados**. Há várias etapas no processo de preparação de dados. Você pode precisar reunir dados e normalizá-los se vierem de fontes diversas. Você pode melhorar a qualidade e a quantidade dos dados por meio de vários métodos, como converter strings em números (como fazemos em [Clustering](../../5-Clustering/1-Visualize/README.md)). Você também pode gerar novos dados com base nos originais (como fazemos em [Classificação](../../4-Classification/1-Introduction/README.md)). Você pode limpar e editar os dados (como faremos antes da lição de [Aplicativo Web](../../3-Web-App/README.md)). Por fim, pode ser necessário randomizar e embaralhar os dados, dependendo das técnicas de treinamento.

✅ Após coletar e processar seus dados, reserve um momento para verificar se sua estrutura permitirá que você responda à pergunta pretendida. Pode ser que os dados não funcionem bem na tarefa proposta, como descobrimos em nossas lições de [Clustering](../../5-Clustering/1-Visualize/README.md)!

### Features e Target

Uma [feature](https://www.datasciencecentral.com/profiles/blogs/an-introduction-to-variable-and-feature-selection) é uma propriedade mensurável dos seus dados. Em muitos conjuntos de dados, ela é expressa como um cabeçalho de coluna, como 'data', 'tamanho' ou 'cor'. Sua variável de feature, geralmente representada como `X` no código, representa a variável de entrada que será usada para treinar o modelo.

Um target é aquilo que você está tentando prever. O target, geralmente representado como `y` no código, representa a resposta à pergunta que você está tentando fazer com seus dados: em dezembro, qual **cor** de abóbora será mais barata? Em São Francisco, quais bairros terão o melhor **preço** de imóveis? Às vezes, o target também é chamado de atributo de rótulo.

### Selecionando sua variável de feature

🎓 **Seleção de Features e Extração de Features** Como saber qual variável escolher ao construir um modelo? Você provavelmente passará por um processo de seleção ou extração de features para escolher as variáveis certas para o modelo mais eficiente. No entanto, elas não são a mesma coisa: "A extração de features cria novas features a partir de funções das features originais, enquanto a seleção de features retorna um subconjunto das features." ([fonte](https://wikipedia.org/wiki/Feature_selection))

### Visualizar seus dados

Um aspecto importante do kit de ferramentas do cientista de dados é o poder de visualizar dados usando várias bibliotecas excelentes, como Seaborn ou MatPlotLib. Representar seus dados visualmente pode permitir que você descubra correlações ocultas que pode aproveitar. Suas visualizações também podem ajudar a identificar vieses ou dados desbalanceados (como descobrimos em [Classificação](../../4-Classification/2-Classifiers-1/README.md)).

### Dividir seu conjunto de dados

Antes de treinar, você precisa dividir seu conjunto de dados em duas ou mais partes de tamanhos desiguais que ainda representem bem os dados.

- **Treinamento**. Esta parte do conjunto de dados é ajustada ao seu modelo para treiná-lo. Este conjunto constitui a maior parte do conjunto de dados original.
- **Teste**. Um conjunto de teste é um grupo independente de dados, frequentemente extraído dos dados originais, que você usa para confirmar o desempenho do modelo construído.
- **Validação**. Um conjunto de validação é um grupo menor e independente de exemplos que você usa para ajustar os hiperparâmetros ou a arquitetura do modelo para melhorá-lo. Dependendo do tamanho dos seus dados e da pergunta que você está fazendo, pode não ser necessário construir este terceiro conjunto (como observamos em [Previsão de Séries Temporais](../../7-TimeSeries/1-Introduction/README.md)).

## Construindo um modelo

Usando seus dados de treinamento, seu objetivo é construir um modelo, ou uma representação estatística dos seus dados, usando vários algoritmos para **treiná-lo**. Treinar um modelo o expõe aos dados e permite que ele faça suposições sobre padrões percebidos que descobre, valida e aceita ou rejeita.

### Decidir sobre um método de treinamento

Dependendo da sua pergunta e da natureza dos seus dados, você escolherá um método para treiná-lo. Explorando a [documentação do Scikit-learn](https://scikit-learn.org/stable/user_guide.html) - que usamos neste curso - você pode explorar várias maneiras de treinar um modelo. Dependendo da sua experiência, pode ser necessário tentar vários métodos diferentes para construir o melhor modelo. É provável que você passe por um processo em que cientistas de dados avaliam o desempenho de um modelo alimentando-o com dados não vistos, verificando sua precisão, vieses e outros problemas que degradam a qualidade, e selecionando o método de treinamento mais apropriado para a tarefa.

### Treinar um modelo

Com seus dados de treinamento em mãos, você está pronto para 'ajustá-los' e criar um modelo. Você notará que em muitas bibliotecas de ML encontrará o código 'model.fit' - é neste momento que você envia sua variável de feature como um array de valores (geralmente 'X') e uma variável de target (geralmente 'y').

### Avaliar o modelo

Uma vez concluído o processo de treinamento (pode levar muitas iterações, ou 'épocas', para treinar um modelo grande), você poderá avaliar a qualidade do modelo usando dados de teste para medir seu desempenho. Esses dados são um subconjunto dos dados originais que o modelo ainda não analisou. Você pode imprimir uma tabela de métricas sobre a qualidade do modelo.

🎓 **Ajuste do modelo**

No contexto de aprendizado de máquina, ajuste do modelo refere-se à precisão da função subjacente do modelo ao tentar analisar dados com os quais não está familiarizado.

🎓 **Subajuste** e **superajuste** são problemas comuns que degradam a qualidade do modelo, pois ele se ajusta de forma insuficiente ou excessiva. Isso faz com que o modelo faça previsões muito alinhadas ou pouco alinhadas com seus dados de treinamento. Um modelo superajustado prevê os dados de treinamento muito bem porque aprendeu os detalhes e ruídos dos dados excessivamente. Um modelo subajustado não é preciso, pois não consegue analisar com precisão nem seus dados de treinamento nem os dados que ainda não 'viu'.

![modelo superajustado](../../../../1-Introduction/4-techniques-of-ML/images/overfitting.png)
> Infográfico por [Jen Looper](https://twitter.com/jenlooper)

## Ajuste de parâmetros

Depois de concluir o treinamento inicial, observe a qualidade do modelo e considere melhorá-lo ajustando seus 'hiperparâmetros'. Leia mais sobre o processo [na documentação](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters?WT.mc_id=academic-77952-leestott).

## Previsão

Este é o momento em que você pode usar dados completamente novos para testar a precisão do modelo. Em um cenário de ML 'aplicado', onde você está construindo ativos web para usar o modelo em produção, este processo pode envolver coletar entrada do usuário (um clique de botão, por exemplo) para definir uma variável e enviá-la ao modelo para inferência ou avaliação.

Nestes módulos, você descobrirá como usar essas etapas para preparar, construir, testar, avaliar e prever - todos os gestos de um cientista de dados e mais, enquanto avança em sua jornada para se tornar um engenheiro de ML 'full stack'.

---

## 🚀Desafio

Desenhe um fluxograma refletindo as etapas de um profissional de ML. Onde você se vê agora no processo? Onde você prevê que encontrará dificuldades? O que parece fácil para você?

## [Quiz pós-aula](https://ff-quizzes.netlify.app/en/ml/)

## Revisão e Autoestudo

Pesquise online entrevistas com cientistas de dados que discutem seu trabalho diário. Aqui está [uma](https://www.youtube.com/watch?v=Z3IjgbbCEfs).

## Tarefa

[Entrevistar um cientista de dados](assignment.md)

---

**Aviso Legal**:  
Este documento foi traduzido utilizando o serviço de tradução por IA [Co-op Translator](https://github.com/Azure/co-op-translator). Embora nos esforcemos para garantir a precisão, esteja ciente de que traduções automatizadas podem conter erros ou imprecisões. O documento original em seu idioma nativo deve ser considerado a fonte autoritativa. Para informações críticas, recomenda-se a tradução profissional realizada por humanos. Não nos responsabilizamos por quaisquer mal-entendidos ou interpretações equivocadas decorrentes do uso desta tradução.