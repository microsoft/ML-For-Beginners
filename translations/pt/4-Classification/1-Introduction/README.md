<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "76438ce4e5d48982d48f1b55c981caac",
  "translation_date": "2025-09-03T18:13:41+00:00",
  "source_file": "4-Classification/1-Introduction/README.md",
  "language_code": "pt"
}
-->
# Introdução à classificação

Nestes quatro módulos, irá explorar um dos focos fundamentais do machine learning clássico - _classificação_. Vamos utilizar vários algoritmos de classificação com um conjunto de dados sobre as brilhantes culinárias da Ásia e da Índia. Esperamos que esteja com fome!

![só uma pitada!](../../../../translated_images/pinch.1b035ec9ba7e0d408313b551b60c721c9c290b2dd2094115bc87e6ddacd114c9.pt.png)

> Celebre as culinárias pan-asiáticas nestas lições! Imagem de [Jen Looper](https://twitter.com/jenlooper)

A classificação é uma forma de [aprendizagem supervisionada](https://wikipedia.org/wiki/Supervised_learning) que tem muito em comum com as técnicas de regressão. Se o machine learning consiste em prever valores ou nomes para coisas utilizando conjuntos de dados, então a classificação geralmente divide-se em dois grupos: _classificação binária_ e _classificação multiclasse_.

[![Introdução à classificação](https://img.youtube.com/vi/eg8DJYwdMyg/0.jpg)](https://youtu.be/eg8DJYwdMyg "Introdução à classificação")

> 🎥 Clique na imagem acima para assistir a um vídeo: John Guttag, do MIT, apresenta a classificação

Lembre-se:

- **Regressão linear** ajudou-o a prever relações entre variáveis e a fazer previsões precisas sobre onde um novo ponto de dados se encaixaria em relação a essa linha. Por exemplo, poderia prever _qual seria o preço de uma abóbora em setembro vs. dezembro_.
- **Regressão logística** ajudou-o a descobrir "categorias binárias": neste ponto de preço, _esta abóbora é laranja ou não-laranja_?

A classificação utiliza vários algoritmos para determinar outras formas de identificar o rótulo ou a classe de um ponto de dados. Vamos trabalhar com estes dados de culinária para ver se, ao observar um grupo de ingredientes, conseguimos determinar a sua origem culinária.

## [Questionário pré-aula](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/19/)

> ### [Esta lição está disponível em R!](../../../../4-Classification/1-Introduction/solution/R/lesson_10.html)

### Introdução

A classificação é uma das atividades fundamentais para investigadores de machine learning e cientistas de dados. Desde a classificação básica de um valor binário ("este email é spam ou não?") até à classificação e segmentação complexa de imagens utilizando visão computacional, é sempre útil ser capaz de organizar dados em classes e fazer perguntas sobre eles.

Para descrever o processo de forma mais científica, o seu método de classificação cria um modelo preditivo que permite mapear a relação entre variáveis de entrada e variáveis de saída.

![classificação binária vs. multiclasse](../../../../translated_images/binary-multiclass.b56d0c86c81105a697dddd82242c1d11e4d78b7afefea07a44627a0f1111c1a9.pt.png)

> Problemas binários vs. multiclasse para algoritmos de classificação. Infográfico de [Jen Looper](https://twitter.com/jenlooper)

Antes de começar o processo de limpeza dos dados, visualizá-los e prepará-los para as nossas tarefas de ML, vamos aprender um pouco sobre as várias formas como o machine learning pode ser utilizado para classificar dados.

Derivada da [estatística](https://wikipedia.org/wiki/Statistical_classification), a classificação utilizando machine learning clássico usa características, como `smoker`, `weight` e `age`, para determinar a _probabilidade de desenvolver X doença_. Como uma técnica de aprendizagem supervisionada semelhante aos exercícios de regressão que realizou anteriormente, os seus dados estão etiquetados, e os algoritmos de ML utilizam essas etiquetas para classificar e prever classes (ou 'características') de um conjunto de dados e atribuí-las a um grupo ou resultado.

✅ Tire um momento para imaginar um conjunto de dados sobre culinárias. O que um modelo multiclasse seria capaz de responder? O que um modelo binário seria capaz de responder? E se quisesse determinar se uma determinada culinária provavelmente utiliza feno-grego? E se quisesse ver se, dado um saco de compras cheio de anis-estrelado, alcachofras, couve-flor e rábano, conseguiria criar um prato típico indiano?

[![Cestos misteriosos malucos](https://img.youtube.com/vi/GuTeDbaNoEU/0.jpg)](https://youtu.be/GuTeDbaNoEU "Cestos misteriosos malucos")

> 🎥 Clique na imagem acima para assistir a um vídeo. Todo o conceito do programa 'Chopped' é o 'cesto misterioso', onde os chefs têm de fazer um prato com uma escolha aleatória de ingredientes. Certamente um modelo de ML teria ajudado!

## Olá 'classificador'

A pergunta que queremos fazer sobre este conjunto de dados de culinária é, na verdade, uma questão de **multiclasse**, já que temos várias possíveis culinárias nacionais com que trabalhar. Dado um lote de ingredientes, a qual destas muitas classes os dados pertencem?

O Scikit-learn oferece vários algoritmos diferentes para classificar dados, dependendo do tipo de problema que deseja resolver. Nas próximas duas lições, aprenderá sobre alguns desses algoritmos.

## Exercício - limpar e equilibrar os seus dados

A primeira tarefa, antes de começar este projeto, é limpar e **equilibrar** os seus dados para obter melhores resultados. Comece com o ficheiro em branco _notebook.ipynb_ na raiz desta pasta.

A primeira coisa a instalar é o [imblearn](https://imbalanced-learn.org/stable/). Este é um pacote do Scikit-learn que permitirá equilibrar melhor os dados (aprenderá mais sobre esta tarefa em breve).

1. Para instalar `imblearn`, execute `pip install`, assim:

    ```python
    pip install imblearn
    ```

1. Importe os pacotes necessários para importar os seus dados e visualizá-los, e também importe `SMOTE` de `imblearn`.

    ```python
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import numpy as np
    from imblearn.over_sampling import SMOTE
    ```

    Agora está pronto para importar os dados.

1. A próxima tarefa será importar os dados:

    ```python
    df  = pd.read_csv('../data/cuisines.csv')
    ```

   Utilizar `read_csv()` irá ler o conteúdo do ficheiro csv _cusines.csv_ e colocá-lo na variável `df`.

1. Verifique a forma dos dados:

    ```python
    df.head()
    ```

   As primeiras cinco linhas têm este aspeto:

    ```output
    |     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
    | --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
    | 0   | 65         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 1   | 66         | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 2   | 67         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 3   | 68         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 4   | 69         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
    ```

1. Obtenha informações sobre estes dados chamando `info()`:

    ```python
    df.info()
    ```

    A sua saída será semelhante a:

    ```output
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2448 entries, 0 to 2447
    Columns: 385 entries, Unnamed: 0 to zucchini
    dtypes: int64(384), object(1)
    memory usage: 7.2+ MB
    ```

## Exercício - aprender sobre culinárias

Agora o trabalho começa a tornar-se mais interessante. Vamos descobrir a distribuição dos dados por culinária.

1. Plote os dados como barras chamando `barh()`:

    ```python
    df.cuisine.value_counts().plot.barh()
    ```

    ![distribuição de dados de culinária](../../../../translated_images/cuisine-dist.d0cc2d551abe5c25f83d73a5f560927e4a061e9a4560bac1e97d35682ef3ca6d.pt.png)

    Existem um número finito de culinárias, mas a distribuição dos dados é desigual. Pode corrigir isso! Antes de o fazer, explore um pouco mais.

1. Descubra a quantidade de dados disponível por culinária e imprima:

    ```python
    thai_df = df[(df.cuisine == "thai")]
    japanese_df = df[(df.cuisine == "japanese")]
    chinese_df = df[(df.cuisine == "chinese")]
    indian_df = df[(df.cuisine == "indian")]
    korean_df = df[(df.cuisine == "korean")]
    
    print(f'thai df: {thai_df.shape}')
    print(f'japanese df: {japanese_df.shape}')
    print(f'chinese df: {chinese_df.shape}')
    print(f'indian df: {indian_df.shape}')
    print(f'korean df: {korean_df.shape}')
    ```

    A saída será semelhante a:

    ```output
    thai df: (289, 385)
    japanese df: (320, 385)
    chinese df: (442, 385)
    indian df: (598, 385)
    korean df: (799, 385)
    ```

## Descobrindo ingredientes

Agora pode aprofundar os dados e aprender quais são os ingredientes típicos por culinária. Deve eliminar dados recorrentes que criam confusão entre culinárias, então vamos aprender sobre este problema.

1. Crie uma função `create_ingredient()` em Python para criar um dataframe de ingredientes. Esta função começará por eliminar uma coluna inútil e organizar os ingredientes pela sua contagem:

    ```python
    def create_ingredient_df(df):
        ingredient_df = df.T.drop(['cuisine','Unnamed: 0']).sum(axis=1).to_frame('value')
        ingredient_df = ingredient_df[(ingredient_df.T != 0).any()]
        ingredient_df = ingredient_df.sort_values(by='value', ascending=False,
        inplace=False)
        return ingredient_df
    ```

   Agora pode usar essa função para ter uma ideia dos dez ingredientes mais populares por culinária.

1. Chame `create_ingredient()` e plote chamando `barh()`:

    ```python
    thai_ingredient_df = create_ingredient_df(thai_df)
    thai_ingredient_df.head(10).plot.barh()
    ```

    ![thai](../../../../translated_images/thai.0269dbab2e78bd38a132067759fe980008bdb80b6d778e5313448dbe12bed846.pt.png)

1. Faça o mesmo para os dados japoneses:

    ```python
    japanese_ingredient_df = create_ingredient_df(japanese_df)
    japanese_ingredient_df.head(10).plot.barh()
    ```

    ![japanese](../../../../translated_images/japanese.30260486f2a05c463c8faa62ebe7b38f0961ed293bd9a6db8eef5d3f0cf17155.pt.png)

1. Agora para os ingredientes chineses:

    ```python
    chinese_ingredient_df = create_ingredient_df(chinese_df)
    chinese_ingredient_df.head(10).plot.barh()
    ```

    ![chinese](../../../../translated_images/chinese.e62cafa5309f111afd1b54490336daf4e927ce32bed837069a0b7ce481dfae8d.pt.png)

1. Plote os ingredientes indianos:

    ```python
    indian_ingredient_df = create_ingredient_df(indian_df)
    indian_ingredient_df.head(10).plot.barh()
    ```

    ![indian](../../../../translated_images/indian.2c4292002af1a1f97a4a24fec6b1459ee8ff616c3822ae56bb62b9903e192af6.pt.png)

1. Finalmente, plote os ingredientes coreanos:

    ```python
    korean_ingredient_df = create_ingredient_df(korean_df)
    korean_ingredient_df.head(10).plot.barh()
    ```

    ![korean](../../../../translated_images/korean.4a4f0274f3d9805a65e61f05597eeaad8620b03be23a2c0a705c023f65fad2c0.pt.png)

1. Agora, elimine os ingredientes mais comuns que criam confusão entre culinárias distintas, chamando `drop()`:

   Toda a gente adora arroz, alho e gengibre!

    ```python
    feature_df= df.drop(['cuisine','Unnamed: 0','rice','garlic','ginger'], axis=1)
    labels_df = df.cuisine #.unique()
    feature_df.head()
    ```

## Equilibrar o conjunto de dados

Agora que limpou os dados, utilize o [SMOTE](https://imbalanced-learn.org/dev/references/generated/imblearn.over_sampling.SMOTE.html) - "Técnica de Sobreamostragem de Minoria Sintética" - para equilibrá-los.

1. Chame `fit_resample()`, esta estratégia gera novas amostras por interpolação.

    ```python
    oversample = SMOTE()
    transformed_feature_df, transformed_label_df = oversample.fit_resample(feature_df, labels_df)
    ```

    Ao equilibrar os seus dados, obterá melhores resultados ao classificá-los. Pense numa classificação binária. Se a maioria dos seus dados pertence a uma classe, um modelo de ML irá prever essa classe com mais frequência, simplesmente porque há mais dados para ela. O equilíbrio dos dados elimina esta distorção.

1. Agora pode verificar os números de etiquetas por ingrediente:

    ```python
    print(f'new label count: {transformed_label_df.value_counts()}')
    print(f'old label count: {df.cuisine.value_counts()}')
    ```

    A sua saída será semelhante a:

    ```output
    new label count: korean      799
    chinese     799
    indian      799
    japanese    799
    thai        799
    Name: cuisine, dtype: int64
    old label count: korean      799
    indian      598
    chinese     442
    japanese    320
    thai        289
    Name: cuisine, dtype: int64
    ```

    Os dados estão limpos, equilibrados e muito apetitosos!

1. O último passo é guardar os seus dados equilibrados, incluindo etiquetas e características, num novo dataframe que pode ser exportado para um ficheiro:

    ```python
    transformed_df = pd.concat([transformed_label_df,transformed_feature_df],axis=1, join='outer')
    ```

1. Pode dar mais uma olhada nos dados utilizando `transformed_df.head()` e `transformed_df.info()`. Guarde uma cópia destes dados para uso em lições futuras:

    ```python
    transformed_df.head()
    transformed_df.info()
    transformed_df.to_csv("../data/cleaned_cuisines.csv")
    ```

    Este novo CSV pode agora ser encontrado na pasta de dados raiz.

---

## 🚀Desafio

Este currículo contém vários conjuntos de dados interessantes. Explore as pastas `data` e veja se alguma contém conjuntos de dados que seriam apropriados para classificação binária ou multiclasse. Que perguntas faria a este conjunto de dados?

## [Questionário pós-aula](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/20/)

## Revisão e Autoestudo

Explore a API do SMOTE. Para que casos de uso é mais indicado? Que problemas resolve?

## Tarefa 

[Explore métodos de classificação](assignment.md)

---

**Aviso Legal**:  
Este documento foi traduzido utilizando o serviço de tradução por IA [Co-op Translator](https://github.com/Azure/co-op-translator). Embora nos esforcemos para garantir a precisão, é importante ter em conta que traduções automáticas podem conter erros ou imprecisões. O documento original na sua língua nativa deve ser considerado a fonte autoritária. Para informações críticas, recomenda-se a tradução profissional realizada por humanos. Não nos responsabilizamos por quaisquer mal-entendidos ou interpretações incorretas decorrentes da utilização desta tradução.