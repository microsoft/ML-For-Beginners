# Criar um Aplicativo Web para usar um Modelo ML

Nesta lição, você treinará um modelo ML em um conjunto de dados que está fora deste mundo: _Avistamentos de OVNIs no último século_, provenientes do banco de dados do NUFORC.

Você aprenderá:

- Como "picles" um modelo treinado
- Como usar esse modelo em um aplicativo Flask

Continuaremos a usar notebooks para limpar dados e treinar nosso modelo, mas você pode levar o processo um passo adiante explorando o uso de um modelo "selvagem", por assim dizer: em um aplicativo Web.

Para fazer isso, você precisa construir um aplicativo Web usando Flask.

## [Teste de pré-aula](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/17/)

## Criando um aplicativo

Há várias maneiras de criar aplicativos Web para consumir modelos de aprendizado de máquina. Sua arquitetura da Web pode influenciar a maneira como seu modelo é treinado. Imagine que você está trabalhando em um negócio onde o grupo de ciência de dados treinou um modelo que eles querem que você use em um aplicativo.

### Considerações

Há muitas perguntas que você precisa fazer:

- **É um aplicativo Web ou um aplicativo móvel?** Se você estiver criando um aplicativo móvel ou precisar usar o modelo em um contexto de IoT, poderá usar [TensorFlow Lite](https://www.tensorflow.org/lite/) e usar o modelo em um aplicativo Android ou iOS.
- **Onde o modelo residirá?** Na nuvem ou localmente?
- **Suporte off-line.** O aplicativo precisa trabalhar off-line?
- **Que tecnologia foi usada para treinar o modelo?** A tecnologia escolhida pode influenciar as ferramentas que você precisa usar.
  
   - **Usando fluxo de Tensor.** Se você estiver treinando um modelo usando TensorFlow, por exemplo, esse ecossistema oferece a capacidade de converter um modelo TensorFlow para uso em um aplicativo Web usando [TensorFlow.js](https://www.tensorflow.org/js/).
- **Usando o PyTorch.** Se você estiver criando um modelo usando uma biblioteca como [PyTorch](https://pytorch.org/), terá a opção de exportá-lo no formato [ONNX](https://onnx.ai/) (Open Neural Network Exchange) para uso em aplicativos Web JavaScript que podem usar o [Onnx Runtime](https://www.onnxruntime.ai/). Essa opção será explorada em uma lição futura para um modelo treinado com o Scikit.
- **Usando o Lobe.ai ou o Azure Custom Vision.** Se você estiver usando um sistema ML SaaS (Software as a Service) como [Lobe.ai](https://lobe.ai/) ou [Azure Custom Vision](https://azure.microsoft.com/services/cognitive-services/custom-vision-service/?WT.mc_id=academy-15963-cxa) para treinar um modelo, esse tipo de software fornece maneiras de exportar o modelo para várias plataformas, incluindo a criação de uma API sob medida ser consultado na nuvem pelo aplicativo online.


Você também tem a oportunidade de construir um aplicativo web Flask inteiro que seria capaz de treinar o próprio modelo em um navegador da web. Isso também pode ser feito usando TensorFlow.js em um contexto JavaScript.

Para nossos propósitos, já que estamos trabalhando com notebooks baseados em Python, vamos explorar as etapas que você precisa seguir para exportar um modelo treinado de tal notebook para um formato legível por um aplicativo web construído em Python.

## Ferramenta

Para esta tarefa, você precisa de duas ferramentas: Flask e Pickle, ambos em Python.

O que é [Frasco](https://palletsprojects.com/p/flask/)? Definido como um 'microframework' por seus criadores, o Flask fornece as características básicas de frameworks web usando Python e um motor de modelagem para construir páginas web. Dê uma olhada em [este módulo de aprendizado](https://docs.microsoft.com/learn/modules/python-flask-build-ai-web-app?WT.mc_id=academic-77952-leestott) para praticar a construção com o Flask.

✅ O que é [Pickle](https://docs.python.org/3/library/pickle.html)? Pickle 🥒 é um módulo Python que serializa e desserializa uma estrutura de objeto Python. Ao "pichar" um modelo, você serializa ou achata sua estrutura para uso na web. Tenha cuidado: o pickle não é intrinsecamente seguro, portanto, tenha cuidado se for solicitado a `cancelar o pickle` de um arquivo. Um arquivo em conserto tem o sufixo `.pkl`.

## Exercício - limpar seus dados

Nesta lição, você usará dados de 80.000 avistamentos de UFO, coletados pelo [NUFORC](https://nuforc.org) (Centro Nacional de Relatórios de UFO). Estes dados têm algumas descrições interessantes de avistamentos de UFO, por exemplo:

- **Descrição de exemplo longo.** "Um homem emerge de um feixe de luz que brilha em um campo gramado à noite e corre em direção ao estacionamento da Texas Instruments".
- **Breve descrição do exemplo.** "as luzes nos perseguiram".

A planilha [ufos.csv](./data/ufos.csv) inclui colunas sobre `city`, `state` e `country` onde ocorreu o avistamento, `shape` do objeto e sua `latitude` e `longitude`.

No espaço em branco [notebook](notebook.ipynb) incluído nesta lição:

1. importe `pandas`, `matplotlib` e `numpy` como fez nas lições anteriores e importe a planilha ufos. Você pode dar uma olhada em um conjunto de dados de amostra:

    ```python
    import pandas as pd
    import numpy as np
    
    ufos = pd.read_csv('./data/ufos.csv')
    ufos.head()
    ```

1.Converta os dados ufos em um pequeno dataframe com títulos novos. Verifique os valores exclusivos no campo `País`.

    ```python
    ufos = pd.DataFrame({'Seconds': ufos['duration (seconds)'], 'Country': ufos['country'],'Latitude': ufos['latitude'],'Longitude': ufos['longitude']})
    
    ufos.Country.unique()
    ```

1. Agora, você pode reduzir a quantidade de dados que precisamos lidar, eliminando quaisquer valores nulos e importando apenas avistamentos entre 1-60 segundos:

    ```python
    ufos.dropna(inplace=True)
    
    ufos = ufos[(ufos['Seconds'] >= 1) & (ufos['Seconds'] <= 60)]
    
    ufos.info()
    ```

1. Importe a biblioteca 'LabelEncoder' do Scikit-learn para converter os valores de texto dos países em um número:

  ✅ LabelEncoder codifica os dados em ordem alfabética

    ```python
    from sklearn.preprocessing import LabelEncoder
    
    ufos['Country'] = LabelEncoder().fit_transform(ufos['Country'])
    
    ufos.head()
    ```

    Seus dados devem ter esta aparência:

    ```output
    	Seconds	Country	Latitude	Longitude
    2	20.0	3		53.200000	-2.916667
    3	20.0	4		28.978333	-96.645833
    14	30.0	4		35.823889	-80.253611
    23	60.0	4		45.582778	-122.352222
    24	3.0		3		51.783333	-0.783333
    ```

## Exercício - construa seu modelo

Agora você pode se preparar para treinar um modelo dividindo os dados no grupo de treinamento e teste.

1. Selecione os três recursos que você deseja treinar como seu vetor X, e o vetor y será o `País`. Você quer digitar `Segundos`, `Latitude` e `Longitude` e obter um ID de país para retornar.

    ```python
    from sklearn.model_selection import train_test_split
    
    Selected_features = ['Seconds','Latitude','Longitude']
    
    X = ufos[Selected_features]
    y = ufos['Country']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    ```

1. Treine seu modelo usando regressão logística:

    ```python
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    print(classification_report(y_test, predictions))
    print('Predicted labels: ', predictions)
    print('Accuracy: ', accuracy_score(y_test, predictions))
    ```

A precisão não é ruim **(cerca de 95%)**, sem surpresas, já que `País` e `Latitude/Longitude` se correlacionam.

O modelo que você criou não é muito revolucionário, pois você deve ser capaz de inferir um `País` a partir de sua `Latitude` e `Longitude`, mas é um bom exercício para tentar treinar a partir de dados brutos que você limpou, exportou e, em seguida, usar esse modelo em um aplicativo Web.

## Exercício - 'pickle' seu modelo

Agora, é hora de _pickle_ seu modelo! Você pode fazer isso em algumas linhas de código. Uma vez que seja _pickled_, carregue seu modelo em pickles e teste-o em uma matriz de dados de amostra contendo valores para segundos, latitude e longitude,

```python
import pickle
model_filename = 'ufo-model.pkl'
pickle.dump(model, open(model_filename,'wb'))

model = pickle.load(open('ufo-model.pkl','rb'))
print(model.predict([[50,44,-12]]))
```

O modelo retorna **'3'**, que é o código de país do Reino Unido. Selvagem! 👽

## Exercício - criar um aplicativo Flask

Agora você pode construir um aplicativo Flask para chamar seu modelo e retornar resultados semelhantes, mas de uma forma mais visualmente agradável.

1. Comece criando uma pasta chamada **web-app** ao lado do arquivo _notebook.ipynb_ onde reside seu arquivo _ufo-model.pkl_.

1. Nessa pasta, crie mais três pastas: **static**, com uma pasta **css** dentro dela e **templates**. Agora você deve ter os seguintes arquivos e diretórios:

    ```output
    web-app/
      static/
        css/
      templates/
    notebook.ipynb
    ufo-model.pkl
    ```

    ✅ Consulte a pasta da solução para obter uma exibição do aplicativo concluído

1. O primeiro arquivo a ser criado na pasta _web-app_ é o arquivo **requirements.txt**. Como _package.json_ em um aplicativo JavaScript, esse arquivo lista as dependências exigidas pelo aplicativo. Em **requirements.txt** adicione as linhas:

    ```text
    scikit-learn
    pandas
    numpy
    flask
    ```

1. Agora, execute este arquivo navegando para _web-app_:

    ```bash
    cd web-app
    ```

1. Em seu terminal, digite `pip install` para instalar as bibliotecas listadas em _requirements.txt_:

    ```bash
    pip install -r requirements.txt
    ```

1.Agora, você está pronto para criar mais três arquivos para concluir o aplicativo:

1. Crie **app.py** na raiz.
2. Crie **index.html** no diretório _templates_.
3. Crie **styles.css** no diretório _static/css_.

1. Construa o arquivo _styles.css_ com alguns estilos:

    ```css
    body {
    	width: 100%;
    	height: 100%;
    	font-family: 'Helvetica';
    	background: black;
    	color: #fff;
    	text-align: center;
    	letter-spacing: 1.4px;
    	font-size: 30px;
    }
    
    input {
    	min-width: 150px;
    }
    
    .grid {
    	width: 300px;
    	border: 1px solid #2d2d2d;
    	display: grid;
    	justify-content: center;
    	margin: 20px auto;
    }
    
    .box {
    	color: #fff;
    	background: #2d2d2d;
    	padding: 12px;
    	display: inline-block;
    }
    ```

1. Em seguida, crie o arquivo _index.html_:

    ```html
    <!DOCTYPE html>
    <html>
      <head>
        <meta charset="UTF-8">
        <title>🛸 UFO Appearance Prediction! 👽</title>
        <link rel="stylesheet" href="{{ url_for('static', filename='css/styles.css') }}">
      </head>
    
      <body>
        <div class="grid">
    
          <div class="box">
    
            <p>According to the number of seconds, latitude and longitude, which country is likely to have reported seeing a UFO?</p>
    
            <form action="{{ url_for('predict')}}" method="post">
              <input type="number" name="seconds" placeholder="Seconds" required="required" min="0" max="60" />
              <input type="text" name="latitude" placeholder="Latitude" required="required" />
              <input type="text" name="longitude" placeholder="Longitude" required="required" />
              <button type="submit" class="btn">Predict country where the UFO is seen</button>
            </form>
    
            <p>{{ prediction_text }}</p>
    
          </div>
    
        </div>
    
      </body>
    </html>
    ```

   Dê uma olhada na modelagem neste arquivo. Observe a sintaxe do 'bigode' ao redor das variáveis que serão fornecidas pelo aplicativo, como o texto de previsão: `{{}}`. Há também uma forma que publica uma previsão da rota `/predict`.

Finalmente, você está pronto para construir o arquivo python que direciona o consumo do modelo e a exibição de previsões:

1. Em `app.py` adicione:

    ```python
    import numpy as np
    from flask import Flask, request, render_template
    import pickle
    
    app = Flask(__name__)
    
    model = pickle.load(open("./ufo-model.pkl", "rb"))
    
    
    @app.route("/")
    def home():
        return render_template("index.html")
    
    
    @app.route("/predict", methods=["POST"])
    def predict():
    
        int_features = [int(x) for x in request.form.values()]
        final_features = [np.array(int_features)]
        prediction = model.predict(final_features)
    
        output = prediction[0]
    
        countries = ["Australia", "Canada", "Germany", "UK", "US"]
    
        return render_template(
            "index.html", prediction_text="Likely country: {}".format(countries[output])
        )
    
    
    if __name__ == "__main__":
        app.run(debug=True)
    ```

    >Dica: quando você adiciona [`debug=True`](https://www.askpython.com/python-modules/flask/flask-debug-mode) ao executar o aplicativo Web usando Flask, todas as alterações feitas no aplicativo serão refletidas imediatamente sem a necessidade de reiniciar o servidor. Cuidado! Não habilite este modo em um aplicativo de produção.

Se você executar `python app.py` ou `python3 app.py` - seu servidor Web é iniciado localmente e você pode preencher um formulário curto para obter uma resposta para sua pergunta de gravação sobre onde os OVNIs foram avistados!

Antes de fazer isso, dê uma olhada nas partes de `app.py`:

1. Primeiro, as dependências são carregadas e o aplicativo inicia.
1. Então, o modelo é importado.
1. Então, index.html é renderizado na rota inicial.

Na rota `/predict`, várias coisas acontecem quando o formulário é publicado:

1. As variáveis de formulário são reunidas e convertidas em uma matriz numérica. Eles são então enviados para o modelo e uma previsão é retornada.
2. Os Países que queremos exibir são renderizados novamente como texto legível de seu código de país previsto, e esse valor é enviado de volta para index.html para ser renderizado no modelo.

Usando um modelo desta maneira, com Flask e um modelo em conserva, é relativamente simples. A coisa mais difícil é entender qual é a forma dos dados que devem ser enviados ao modelo para obter uma previsão. Tudo depende de como o modelo foi treinado. Este tem três pontos de dados para serem inseridos a fim de obter uma previsão.

Em um ambiente profissional, você pode ver como uma boa comunicação é necessária entre as pessoas que treinam o modelo e aqueles que o consomem em um aplicativo web ou móvel. No nosso caso, é só uma pessoa, você!

---

## 🚀Desafio

Em vez de trabalhar em um notebook e importar o modelo para o aplicativo Flask, você poderia treinar o modelo dentro do aplicativo Flask! Tente converter seu código Python no notebook, talvez depois que seus dados forem limpos, para treinar o modelo de dentro do aplicativo em uma rota chamada `train`. Quais são os prós e contras de se buscar esse método?

## [Teste pós-aula](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/18/)

## Análise e autoestudo

Há muitas maneiras de construir um aplicativo Web para consumir modelos ML. Faça uma lista de maneiras de usar JavaScript ou Python para construir um aplicativo Web para aproveitar o aprendizado de máquina. Considere a arquitetura: o modelo deve permanecer no aplicativo ou viver na nuvem? Se o último, como você acessaria? Desenhe um modelo arquitetônico para uma solução web ML aplicada.

## Atribuição

[Tente um modelo diferente](assignment.md)
