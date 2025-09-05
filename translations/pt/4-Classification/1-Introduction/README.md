<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "aaf391d922bd6de5efba871d514c6d47",
  "translation_date": "2025-09-05T08:48:08+00:00",
  "source_file": "4-Classification/1-Introduction/README.md",
  "language_code": "pt"
}
-->
# Introdução à classificação

Nestes quatro módulos, vais explorar um dos focos fundamentais do machine learning clássico - _classificação_. Vamos utilizar vários algoritmos de classificação com um conjunto de dados sobre as brilhantes culinárias da Ásia e da Índia. Espero que estejas com fome!

![uma pitada!](../../../../4-Classification/1-Introduction/images/pinch.png)

> Celebra as culinárias pan-asiáticas nestas lições! Imagem por [Jen Looper](https://twitter.com/jenlooper)

Classificação é uma forma de [aprendizagem supervisionada](https://wikipedia.org/wiki/Supervised_learning) que tem muito em comum com técnicas de regressão. Se o machine learning consiste em prever valores ou nomes para coisas utilizando conjuntos de dados, então a classificação geralmente divide-se em dois grupos: _classificação binária_ e _classificação multicategorias_.

[![Introdução à classificação](https://img.youtube.com/vi/eg8DJYwdMyg/0.jpg)](https://youtu.be/eg8DJYwdMyg "Introdução à classificação")

> 🎥 Clica na imagem acima para ver um vídeo: John Guttag do MIT apresenta a classificação

Lembra-te:

- **Regressão linear** ajudou-te a prever relações entre variáveis e a fazer previsões precisas sobre onde um novo ponto de dados se encaixaria em relação a essa linha. Por exemplo, podias prever _qual seria o preço de uma abóbora em setembro vs. dezembro_.
- **Regressão logística** ajudou-te a descobrir "categorias binárias": a este preço, _esta abóbora é laranja ou não-laranja_?

A classificação utiliza vários algoritmos para determinar outras formas de identificar o rótulo ou classe de um ponto de dados. Vamos trabalhar com estes dados sobre culinárias para ver se, ao observar um grupo de ingredientes, conseguimos determinar a sua origem culinária.

## [Questionário pré-aula](https://ff-quizzes.netlify.app/en/ml/)

> ### [Esta lição está disponível em R!](../../../../4-Classification/1-Introduction/solution/R/lesson_10.html)

### Introdução

Classificação é uma das atividades fundamentais do investigador de machine learning e do cientista de dados. Desde a classificação básica de um valor binário ("este email é spam ou não?"), até à classificação e segmentação complexa de imagens utilizando visão computacional, é sempre útil conseguir organizar dados em classes e fazer perguntas sobre eles.

De forma mais científica, o teu método de classificação cria um modelo preditivo que te permite mapear a relação entre variáveis de entrada e variáveis de saída.

![classificação binária vs. multicategorias](../../../../4-Classification/1-Introduction/images/binary-multiclass.png)

> Problemas binários vs. multicategorias para algoritmos de classificação. Infográfico por [Jen Looper](https://twitter.com/jenlooper)

Antes de começar o processo de limpeza dos dados, visualizá-los e prepará-los para as nossas tarefas de ML, vamos aprender um pouco sobre as várias formas como o machine learning pode ser utilizado para classificar dados.

Derivada da [estatística](https://wikipedia.org/wiki/Statistical_classification), a classificação utilizando machine learning clássico usa características, como `fumador`, `peso` e `idade`, para determinar _probabilidade de desenvolver X doença_. Como técnica de aprendizagem supervisionada semelhante aos exercícios de regressão que realizaste anteriormente, os teus dados estão rotulados e os algoritmos de ML utilizam esses rótulos para classificar e prever classes (ou 'características') de um conjunto de dados e atribuí-las a um grupo ou resultado.

✅ Tira um momento para imaginar um conjunto de dados sobre culinárias. Que tipo de perguntas um modelo multicategorias poderia responder? E um modelo binário? E se quisesses determinar se uma determinada culinária provavelmente utiliza feno-grego? E se quisesses ver se, dado um saco de compras cheio de anis-estrelado, alcachofras, couve-flor e rábano, conseguirias criar um prato típico indiano?

[![Cestos misteriosos malucos](https://img.youtube.com/vi/GuTeDbaNoEU/0.jpg)](https://youtu.be/GuTeDbaNoEU "Cestos misteriosos malucos")

> 🎥 Clica na imagem acima para ver um vídeo. Todo o conceito do programa 'Chopped' é o 'cesto misterioso', onde os chefs têm de criar um prato com uma escolha aleatória de ingredientes. Certamente um modelo de ML teria ajudado!

## Olá 'classificador'

A pergunta que queremos fazer sobre este conjunto de dados de culinárias é, na verdade, uma **questão multicategorias**, já que temos várias possíveis culinárias nacionais com que trabalhar. Dado um conjunto de ingredientes, a qual destas muitas classes os dados pertencem?

O Scikit-learn oferece vários algoritmos diferentes para classificar dados, dependendo do tipo de problema que queres resolver. Nas próximas duas lições, vais aprender sobre alguns desses algoritmos.

## Exercício - limpar e equilibrar os dados

A primeira tarefa, antes de começar este projeto, é limpar e **equilibrar** os teus dados para obter melhores resultados. Começa com o ficheiro vazio _notebook.ipynb_ na raiz desta pasta.

A primeira coisa a instalar é [imblearn](https://imbalanced-learn.org/stable/). Este é um pacote do Scikit-learn que te permitirá equilibrar melhor os dados (vais aprender mais sobre esta tarefa em breve).

1. Para instalar `imblearn`, executa `pip install`, assim:

    ```python
    pip install imblearn
    ```

1. Importa os pacotes necessários para importar os teus dados e visualizá-los, e também importa `SMOTE` de `imblearn`.

    ```python
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import numpy as np
    from imblearn.over_sampling import SMOTE
    ```

    Agora estás pronto para importar os dados.

1. A próxima tarefa será importar os dados:

    ```python
    df  = pd.read_csv('../data/cuisines.csv')
    ```

   Utilizar `read_csv()` irá ler o conteúdo do ficheiro csv _cusines.csv_ e colocá-lo na variável `df`.

1. Verifica a forma dos dados:

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

1. Obtém informações sobre estes dados chamando `info()`:

    ```python
    df.info()
    ```

    O teu output será semelhante a:

    ```output
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2448 entries, 0 to 2447
    Columns: 385 entries, Unnamed: 0 to zucchini
    dtypes: int64(384), object(1)
    memory usage: 7.2+ MB
    ```

## Exercício - aprender sobre culinárias

Agora o trabalho começa a tornar-se mais interessante. Vamos descobrir a distribuição dos dados, por culinária.

1. Representa os dados como barras chamando `barh()`:

    ```python
    df.cuisine.value_counts().plot.barh()
    ```

    ![distribuição de dados de culinárias](../../../../4-Classification/1-Introduction/images/cuisine-dist.png)

    Existem um número finito de culinárias, mas a distribuição dos dados é desigual. Podes corrigir isso! Antes de o fazer, explora um pouco mais.

1. Descobre a quantidade de dados disponível por culinária e imprime-a:

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

    o output será semelhante a:

    ```output
    thai df: (289, 385)
    japanese df: (320, 385)
    chinese df: (442, 385)
    indian df: (598, 385)
    korean df: (799, 385)
    ```

## Descobrir ingredientes

Agora podes aprofundar os dados e aprender quais são os ingredientes típicos por culinária. Deves limpar dados recorrentes que criam confusão entre culinárias, então vamos aprender sobre este problema.

1. Cria uma função `create_ingredient()` em Python para criar um dataframe de ingredientes. Esta função começará por eliminar uma coluna inútil e organizar os ingredientes pelo seu número:

    ```python
    def create_ingredient_df(df):
        ingredient_df = df.T.drop(['cuisine','Unnamed: 0']).sum(axis=1).to_frame('value')
        ingredient_df = ingredient_df[(ingredient_df.T != 0).any()]
        ingredient_df = ingredient_df.sort_values(by='value', ascending=False,
        inplace=False)
        return ingredient_df
    ```

   Agora podes usar essa função para ter uma ideia dos dez ingredientes mais populares por culinária.

1. Chama `create_ingredient()` e representa os dados chamando `barh()`:

    ```python
    thai_ingredient_df = create_ingredient_df(thai_df)
    thai_ingredient_df.head(10).plot.barh()
    ```

    ![thai](../../../../4-Classification/1-Introduction/images/thai.png)

1. Faz o mesmo para os dados japoneses:

    ```python
    japanese_ingredient_df = create_ingredient_df(japanese_df)
    japanese_ingredient_df.head(10).plot.barh()
    ```

    ![japanese](../../../../4-Classification/1-Introduction/images/japanese.png)

1. Agora para os ingredientes chineses:

    ```python
    chinese_ingredient_df = create_ingredient_df(chinese_df)
    chinese_ingredient_df.head(10).plot.barh()
    ```

    ![chinese](../../../../4-Classification/1-Introduction/images/chinese.png)

1. Representa os ingredientes indianos:

    ```python
    indian_ingredient_df = create_ingredient_df(indian_df)
    indian_ingredient_df.head(10).plot.barh()
    ```

    ![indian](../../../../4-Classification/1-Introduction/images/indian.png)

1. Finalmente, representa os ingredientes coreanos:

    ```python
    korean_ingredient_df = create_ingredient_df(korean_df)
    korean_ingredient_df.head(10).plot.barh()
    ```

    ![korean](../../../../4-Classification/1-Introduction/images/korean.png)

1. Agora, elimina os ingredientes mais comuns que criam confusão entre culinárias distintas, chamando `drop()`:

   Toda a gente adora arroz, alho e gengibre!

    ```python
    feature_df= df.drop(['cuisine','Unnamed: 0','rice','garlic','ginger'], axis=1)
    labels_df = df.cuisine #.unique()
    feature_df.head()
    ```

## Equilibrar o conjunto de dados

Agora que limpaste os dados, utiliza [SMOTE](https://imbalanced-learn.org/dev/references/generated/imblearn.over_sampling.SMOTE.html) - "Técnica de Sobreamostragem Sintética de Minorias" - para equilibrá-los.

1. Chama `fit_resample()`, esta estratégia gera novas amostras por interpolação.

    ```python
    oversample = SMOTE()
    transformed_feature_df, transformed_label_df = oversample.fit_resample(feature_df, labels_df)
    ```

    Ao equilibrar os teus dados, vais obter melhores resultados ao classificá-los. Pensa numa classificação binária. Se a maior parte dos teus dados pertence a uma classe, um modelo de ML vai prever essa classe com mais frequência, apenas porque há mais dados para ela. O equilíbrio dos dados elimina esta desigualdade.

1. Agora podes verificar os números de rótulos por ingrediente:

    ```python
    print(f'new label count: {transformed_label_df.value_counts()}')
    print(f'old label count: {df.cuisine.value_counts()}')
    ```

    O teu output será semelhante a:

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

    Os dados estão limpos, equilibrados e muito deliciosos!

1. O último passo é guardar os teus dados equilibrados, incluindo rótulos e características, num novo dataframe que pode ser exportado para um ficheiro:

    ```python
    transformed_df = pd.concat([transformed_label_df,transformed_feature_df],axis=1, join='outer')
    ```

1. Podes dar mais uma olhada nos dados utilizando `transformed_df.head()` e `transformed_df.info()`. Guarda uma cópia destes dados para uso em lições futuras:

    ```python
    transformed_df.head()
    transformed_df.info()
    transformed_df.to_csv("../data/cleaned_cuisines.csv")
    ```

    Este novo CSV pode agora ser encontrado na pasta de dados raiz.

---

## 🚀Desafio

Este currículo contém vários conjuntos de dados interessantes. Explora as pastas `data` e vê se alguma contém conjuntos de dados que seriam apropriados para classificação binária ou multicategorias. Que perguntas farias a este conjunto de dados?

## [Questionário pós-aula](https://ff-quizzes.netlify.app/en/ml/)

## Revisão & Estudo Individual

Explora a API do SMOTE. Para que casos de uso é mais adequado? Que problemas resolve?

## Tarefa 

[Explora métodos de classificação](assignment.md)

---

**Aviso Legal**:  
Este documento foi traduzido utilizando o serviço de tradução automática [Co-op Translator](https://github.com/Azure/co-op-translator). Embora nos esforcemos para garantir a precisão, esteja ciente de que traduções automáticas podem conter erros ou imprecisões. O documento original na sua língua nativa deve ser considerado a fonte oficial. Para informações críticas, recomenda-se uma tradução profissional realizada por humanos. Não nos responsabilizamos por quaisquer mal-entendidos ou interpretações incorretas resultantes do uso desta tradução.