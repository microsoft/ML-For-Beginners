<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "6b1cb0e46d4c5b747eff6e3607642760",
  "translation_date": "2025-09-03T16:37:29+00:00",
  "source_file": "2-Regression/1-Tools/README.md",
  "language_code": "pt"
}
-->
# Introdução ao Python e Scikit-learn para modelos de regressão

![Resumo de regressões em um sketchnote](../../../../translated_images/ml-regression.4e4f70e3b3ed446e3ace348dec973e133fa5d3680fbc8412b61879507369b98d.pt.png)

> Sketchnote por [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Questionário pré-aula](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/9/)

> ### [Esta aula está disponível em R!](../../../../2-Regression/1-Tools/solution/R/lesson_1.html)

## Introdução

Nestes quatro módulos, vais aprender como construir modelos de regressão. Vamos discutir para que servem em breve. Mas antes de começar, certifica-te de que tens as ferramentas certas para iniciar o processo!

Nesta aula, vais aprender a:

- Configurar o teu computador para tarefas locais de aprendizagem automática.
- Trabalhar com Jupyter notebooks.
- Utilizar Scikit-learn, incluindo a instalação.
- Explorar regressão linear com um exercício prático.

## Instalações e configurações

[![ML para iniciantes - Configura as tuas ferramentas para criar modelos de Machine Learning](https://img.youtube.com/vi/-DfeD2k2Kj0/0.jpg)](https://youtu.be/-DfeD2k2Kj0 "ML para iniciantes - Configura as tuas ferramentas para criar modelos de Machine Learning")

> 🎥 Clica na imagem acima para um vídeo curto sobre como configurar o teu computador para ML.

1. **Instalar Python**. Certifica-te de que [Python](https://www.python.org/downloads/) está instalado no teu computador. Vais usar Python para muitas tarefas de ciência de dados e aprendizagem automática. A maioria dos sistemas já inclui uma instalação de Python. Existem também [Pacotes de Codificação Python](https://code.visualstudio.com/learn/educators/installers?WT.mc_id=academic-77952-leestott) úteis para facilitar a configuração para alguns utilizadores.

   Algumas utilizações de Python, no entanto, requerem uma versão específica do software. Por isso, é útil trabalhar num [ambiente virtual](https://docs.python.org/3/library/venv.html).

2. **Instalar Visual Studio Code**. Certifica-te de que tens o Visual Studio Code instalado no teu computador. Segue estas instruções para [instalar o Visual Studio Code](https://code.visualstudio.com/) para uma instalação básica. Vais usar Python no Visual Studio Code neste curso, por isso pode ser útil rever como [configurar o Visual Studio Code](https://docs.microsoft.com/learn/modules/python-install-vscode?WT.mc_id=academic-77952-leestott) para desenvolvimento em Python.

   > Familiariza-te com Python ao explorar esta coleção de [módulos de aprendizagem](https://docs.microsoft.com/users/jenlooper-2911/collections/mp1pagggd5qrq7?WT.mc_id=academic-77952-leestott)
   >
   > [![Configurar Python com Visual Studio Code](https://img.youtube.com/vi/yyQM70vi7V8/0.jpg)](https://youtu.be/yyQM70vi7V8 "Configurar Python com Visual Studio Code")
   >
   > 🎥 Clica na imagem acima para um vídeo: usar Python no VS Code.

3. **Instalar Scikit-learn**, seguindo [estas instruções](https://scikit-learn.org/stable/install.html). Como precisas de garantir que usas Python 3, é recomendado que uses um ambiente virtual. Nota que, se estiveres a instalar esta biblioteca num Mac M1, há instruções especiais na página acima.

4. **Instalar Jupyter Notebook**. Vais precisar de [instalar o pacote Jupyter](https://pypi.org/project/jupyter/).

## O teu ambiente de autoria de ML

Vais usar **notebooks** para desenvolver o teu código Python e criar modelos de aprendizagem automática. Este tipo de ficheiro é uma ferramenta comum para cientistas de dados e pode ser identificado pela extensão `.ipynb`.

Os notebooks são um ambiente interativo que permite ao programador tanto codificar como adicionar notas e escrever documentação em torno do código, o que é bastante útil para projetos experimentais ou orientados à pesquisa.

[![ML para iniciantes - Configurar Jupyter Notebooks para começar a construir modelos de regressão](https://img.youtube.com/vi/7E-jC8FLA2E/0.jpg)](https://youtu.be/7E-jC8FLA2E "ML para iniciantes - Configurar Jupyter Notebooks para começar a construir modelos de regressão")

> 🎥 Clica na imagem acima para um vídeo curto sobre este exercício.

### Exercício - trabalhar com um notebook

Nesta pasta, vais encontrar o ficheiro _notebook.ipynb_.

1. Abre _notebook.ipynb_ no Visual Studio Code.

   Um servidor Jupyter será iniciado com Python 3+. Vais encontrar áreas do notebook que podem ser `executadas`, pedaços de código. Podes executar um bloco de código ao selecionar o ícone que parece um botão de reprodução.

2. Seleciona o ícone `md` e adiciona um pouco de markdown, com o seguinte texto **# Bem-vindo ao teu notebook**.

   Em seguida, adiciona algum código Python.

3. Escreve **print('hello notebook')** no bloco de código.
4. Seleciona a seta para executar o código.

   Deverás ver a seguinte declaração impressa:

    ```output
    hello notebook
    ```

![VS Code com um notebook aberto](../../../../translated_images/notebook.4a3ee31f396b88325607afda33cadcc6368de98040ff33942424260aa84d75f2.pt.jpg)

Podes intercalar o teu código com comentários para auto-documentar o notebook.

✅ Pensa por um momento como o ambiente de trabalho de um programador web é diferente do de um cientista de dados.

## Começar com Scikit-learn

Agora que o Python está configurado no teu ambiente local e estás confortável com Jupyter notebooks, vamos ficar igualmente confortáveis com Scikit-learn (pronuncia-se `sci` como em `science`). Scikit-learn fornece uma [API extensa](https://scikit-learn.org/stable/modules/classes.html#api-ref) para ajudar-te a realizar tarefas de ML.

De acordo com o [site deles](https://scikit-learn.org/stable/getting_started.html), "Scikit-learn é uma biblioteca de aprendizagem automática de código aberto que suporta aprendizagem supervisionada e não supervisionada. Também fornece várias ferramentas para ajuste de modelos, pré-processamento de dados, seleção e avaliação de modelos, e muitas outras utilidades."

Neste curso, vais usar Scikit-learn e outras ferramentas para construir modelos de aprendizagem automática para realizar o que chamamos de tarefas de 'aprendizagem automática tradicional'. Evitamos deliberadamente redes neurais e aprendizagem profunda, pois são melhor abordadas no nosso próximo currículo 'AI para Iniciantes'.

Scikit-learn torna simples construir modelos e avaliá-los para uso. Está principalmente focado em usar dados numéricos e contém vários conjuntos de dados prontos para uso como ferramentas de aprendizagem. Também inclui modelos pré-construídos para os alunos experimentarem. Vamos explorar o processo de carregar dados pré-embalados e usar um estimador para o primeiro modelo de ML com Scikit-learn com alguns dados básicos.

## Exercício - o teu primeiro notebook com Scikit-learn

> Este tutorial foi inspirado pelo [exemplo de regressão linear](https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py) no site do Scikit-learn.

[![ML para iniciantes - O teu primeiro projeto de regressão linear em Python](https://img.youtube.com/vi/2xkXL5EUpS0/0.jpg)](https://youtu.be/2xkXL5EUpS0 "ML para iniciantes - O teu primeiro projeto de regressão linear em Python")

> 🎥 Clica na imagem acima para um vídeo curto sobre este exercício.

No ficheiro _notebook.ipynb_ associado a esta aula, limpa todas as células pressionando o ícone 'lixeira'.

Nesta secção, vais trabalhar com um pequeno conjunto de dados sobre diabetes que está incluído no Scikit-learn para fins de aprendizagem. Imagina que queres testar um tratamento para pacientes diabéticos. Modelos de aprendizagem automática podem ajudar-te a determinar quais pacientes responderiam melhor ao tratamento, com base em combinações de variáveis. Mesmo um modelo de regressão muito básico, quando visualizado, pode mostrar informações sobre variáveis que ajudariam a organizar os teus ensaios clínicos teóricos.

✅ Existem muitos tipos de métodos de regressão, e qual escolher depende da resposta que procuras. Se quiseres prever a altura provável de uma pessoa com uma determinada idade, usarias regressão linear, pois estás a procurar um **valor numérico**. Se estiveres interessado em descobrir se um tipo de cozinha deve ser considerado vegan ou não, estás a procurar uma **atribuição de categoria**, então usarias regressão logística. Vais aprender mais sobre regressão logística mais tarde. Pensa um pouco sobre algumas perguntas que podes fazer aos dados e qual destes métodos seria mais apropriado.

Vamos começar esta tarefa.

### Importar bibliotecas

Para esta tarefa, vamos importar algumas bibliotecas:

- **matplotlib**. É uma ferramenta útil para [criação de gráficos](https://matplotlib.org/) e vamos usá-la para criar um gráfico de linha.
- **numpy**. [numpy](https://numpy.org/doc/stable/user/whatisnumpy.html) é uma biblioteca útil para lidar com dados numéricos em Python.
- **sklearn**. Esta é a biblioteca [Scikit-learn](https://scikit-learn.org/stable/user_guide.html).

Importa algumas bibliotecas para ajudar nas tuas tarefas.

1. Adiciona as importações escrevendo o seguinte código:

   ```python
   import matplotlib.pyplot as plt
   import numpy as np
   from sklearn import datasets, linear_model, model_selection
   ```

   Acima, estás a importar `matplotlib`, `numpy` e estás a importar `datasets`, `linear_model` e `model_selection` de `sklearn`. `model_selection` é usado para dividir dados em conjuntos de treino e teste.

### O conjunto de dados sobre diabetes

O [conjunto de dados sobre diabetes](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) incluído contém 442 amostras de dados sobre diabetes, com 10 variáveis de características, algumas das quais incluem:

- age: idade em anos
- bmi: índice de massa corporal
- bp: pressão arterial média
- s1 tc: células T (um tipo de glóbulos brancos)

✅ Este conjunto de dados inclui o conceito de 'sexo' como uma variável de característica importante para pesquisas sobre diabetes. Muitos conjuntos de dados médicos incluem este tipo de classificação binária. Pensa um pouco sobre como categorizações como esta podem excluir certas partes da população de tratamentos.

Agora, carrega os dados X e y.

> 🎓 Lembra-te, isto é aprendizagem supervisionada, e precisamos de um alvo 'y' nomeado.

Numa nova célula de código, carrega o conjunto de dados sobre diabetes chamando `load_diabetes()`. O input `return_X_y=True` indica que `X` será uma matriz de dados e `y` será o alvo da regressão.

1. Adiciona alguns comandos de impressão para mostrar a forma da matriz de dados e o seu primeiro elemento:

    ```python
    X, y = datasets.load_diabetes(return_X_y=True)
    print(X.shape)
    print(X[0])
    ```

    O que estás a obter como resposta é uma tupla. O que estás a fazer é atribuir os dois primeiros valores da tupla a `X` e `y`, respetivamente. Aprende mais [sobre tuplas](https://wikipedia.org/wiki/Tuple).

    Podes ver que estes dados têm 442 itens organizados em arrays de 10 elementos:

    ```text
    (442, 10)
    [ 0.03807591  0.05068012  0.06169621  0.02187235 -0.0442235  -0.03482076
    -0.04340085 -0.00259226  0.01990842 -0.01764613]
    ```

    ✅ Pensa um pouco sobre a relação entre os dados e o alvo da regressão. A regressão linear prevê relações entre a característica X e a variável alvo y. Consegues encontrar o [alvo](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) para o conjunto de dados sobre diabetes na documentação? O que este conjunto de dados está a demonstrar, dado o alvo?

2. Em seguida, seleciona uma parte deste conjunto de dados para plotar, escolhendo a 3ª coluna do conjunto de dados. Podes fazer isso usando o operador `:` para selecionar todas as linhas e, depois, selecionar a 3ª coluna usando o índice (2). Também podes remodelar os dados para serem um array 2D - como necessário para plotagem - usando `reshape(n_rows, n_columns)`. Se um dos parâmetros for -1, a dimensão correspondente é calculada automaticamente.

   ```python
   X = X[:, 2]
   X = X.reshape((-1,1))
   ```

   ✅ A qualquer momento, imprime os dados para verificar a sua forma.

3. Agora que tens os dados prontos para serem plotados, podes ver se uma máquina pode ajudar a determinar uma divisão lógica entre os números neste conjunto de dados. Para fazer isso, precisas de dividir tanto os dados (X) quanto o alvo (y) em conjuntos de teste e treino. Scikit-learn tem uma forma simples de fazer isso; podes dividir os teus dados de teste num ponto específico.

   ```python
   X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.33)
   ```

4. Agora estás pronto para treinar o teu modelo! Carrega o modelo de regressão linear e treina-o com os teus conjuntos de treino X e y usando `model.fit()`:

    ```python
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)
    ```

    ✅ `model.fit()` é uma função que vais ver em muitas bibliotecas de ML, como TensorFlow.

5. Em seguida, cria uma previsão usando os dados de teste, com a função `predict()`. Isto será usado para desenhar a linha entre os grupos de dados.

    ```python
    y_pred = model.predict(X_test)
    ```

6. Agora é hora de mostrar os dados num gráfico. Matplotlib é uma ferramenta muito útil para esta tarefa. Cria um gráfico de dispersão de todos os dados de teste X e y e usa a previsão para desenhar uma linha no lugar mais apropriado, entre os agrupamentos de dados do modelo.

    ```python
    plt.scatter(X_test, y_test,  color='black')
    plt.plot(X_test, y_pred, color='blue', linewidth=3)
    plt.xlabel('Scaled BMIs')
    plt.ylabel('Disease Progression')
    plt.title('A Graph Plot Showing Diabetes Progression Against BMI')
    plt.show()
    ```

   ![um gráfico de dispersão mostrando pontos de dados sobre diabetes](../../../../translated_images/scatterplot.ad8b356bcbb33be68d54050e09b9b7bfc03e94fde7371f2609ae43f4c563b2d7.pt.png)
✅ Pensa um pouco sobre o que está a acontecer aqui. Uma linha reta está a passar por muitos pequenos pontos de dados, mas o que está realmente a fazer? Consegues perceber como deverias ser capaz de usar esta linha para prever onde um novo ponto de dados, ainda não visto, deveria encaixar em relação ao eixo y do gráfico? Tenta colocar em palavras a utilidade prática deste modelo.

Parabéns, construíste o teu primeiro modelo de regressão linear, criaste uma previsão com ele e exibiste-a num gráfico!

---
## 🚀Desafio

Representa graficamente uma variável diferente deste conjunto de dados. Dica: edita esta linha: `X = X[:,2]`. Dado o objetivo deste conjunto de dados, o que consegues descobrir sobre a progressão da diabetes como doença?

## [Questionário pós-aula](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/10/)

## Revisão & Autoestudo

Neste tutorial, trabalhaste com regressão linear simples, em vez de regressão univariada ou múltipla. Lê um pouco sobre as diferenças entre estes métodos ou vê [este vídeo](https://www.coursera.org/lecture/quantifying-relationships-regression-models/linear-vs-nonlinear-categorical-variables-ai2Ef).

Lê mais sobre o conceito de regressão e reflete sobre que tipo de perguntas podem ser respondidas com esta técnica. Faz este [tutorial](https://docs.microsoft.com/learn/modules/train-evaluate-regression-models?WT.mc_id=academic-77952-leestott) para aprofundar o teu entendimento.

## Tarefa

[Um conjunto de dados diferente](assignment.md)

---

**Aviso Legal**:  
Este documento foi traduzido utilizando o serviço de tradução por IA [Co-op Translator](https://github.com/Azure/co-op-translator). Embora nos esforcemos para garantir a precisão, é importante ter em conta que traduções automáticas podem conter erros ou imprecisões. O documento original na sua língua nativa deve ser considerado a fonte autoritária. Para informações críticas, recomenda-se a tradução profissional realizada por humanos. Não nos responsabilizamos por quaisquer mal-entendidos ou interpretações incorretas decorrentes da utilização desta tradução.