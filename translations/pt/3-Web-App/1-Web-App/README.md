# Construa um App Web para usar um Modelo de ML

Nesta lição, você irá treinar um modelo de ML em um conjunto de dados que está fora deste mundo: _avistamentos de OVNIs no último século_, extraídos do banco de dados da NUFORC.

Você aprenderá:

- Como 'pickle' um modelo treinado
- Como usar esse modelo em um app Flask

Continuaremos a usar notebooks para limpar os dados e treinar nosso modelo, mas você pode levar o processo um passo adiante explorando como usar um modelo 'no mundo real', por assim dizer: em um app web.

Para fazer isso, você precisa construir um app web usando Flask.

## [Quiz pré-aula](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/17/)

## Construindo um app

Existem várias maneiras de construir apps web para consumir modelos de aprendizado de máquina. Sua arquitetura web pode influenciar a forma como seu modelo é treinado. Imagine que você está trabalhando em uma empresa onde o grupo de ciência de dados treinou um modelo que eles querem que você use em um app.

### Considerações

Há muitas perguntas que você precisa fazer:

- **É um app web ou um app móvel?** Se você estiver construindo um app móvel ou precisar usar o modelo em um contexto de IoT, você poderia usar [TensorFlow Lite](https://www.tensorflow.org/lite/) e usar o modelo em um app Android ou iOS.
- **Onde o modelo residirá?** Na nuvem ou localmente?
- **Suporte offline.** O app precisa funcionar offline?
- **Que tecnologia foi usada para treinar o modelo?** A tecnologia escolhida pode influenciar as ferramentas que você precisa usar.
    - **Usando TensorFlow.** Se você estiver treinando um modelo usando TensorFlow, por exemplo, esse ecossistema fornece a capacidade de converter um modelo TensorFlow para uso em um app web usando [TensorFlow.js](https://www.tensorflow.org/js/).
    - **Usando PyTorch.** Se você estiver construindo um modelo usando uma biblioteca como [PyTorch](https://pytorch.org/), você tem a opção de exportá-lo no formato [ONNX](https://onnx.ai/) (Open Neural Network Exchange) para uso em apps web JavaScript que podem usar o [Onnx Runtime](https://www.onnxruntime.ai/). Esta opção será explorada em uma lição futura para um modelo treinado com Scikit-learn.
    - **Usando Lobe.ai ou Azure Custom Vision.** Se você estiver usando um sistema de ML SaaS (Software como Serviço) como [Lobe.ai](https://lobe.ai/) ou [Azure Custom Vision](https://azure.microsoft.com/services/cognitive-services/custom-vision-service/?WT.mc_id=academic-77952-leestott) para treinar um modelo, esse tipo de software fornece maneiras de exportar o modelo para muitas plataformas, incluindo a construção de uma API sob medida para ser consultada na nuvem pelo seu aplicativo online.

Você também tem a oportunidade de construir um app web Flask completo que seria capaz de treinar o modelo em um navegador web. Isso também pode ser feito usando TensorFlow.js em um contexto JavaScript.

Para nossos propósitos, uma vez que temos trabalhado com notebooks baseados em Python, vamos explorar os passos que você precisa seguir para exportar um modelo treinado de tal notebook para um formato legível por um app web construído em Python.

## Ferramenta

Para esta tarefa, você precisa de duas ferramentas: Flask e Pickle, ambas que rodam em Python.

✅ O que é [Flask](https://palletsprojects.com/p/flask/)? Definido como um 'micro-framework' por seus criadores, o Flask fornece os recursos básicos dos frameworks web usando Python e um motor de templates para construir páginas web. Dê uma olhada neste [módulo Learn](https://docs.microsoft.com/learn/modules/python-flask-build-ai-web-app?WT.mc_id=academic-77952-leestott) para praticar a construção com Flask.

✅ O que é [Pickle](https://docs.python.org/3/library/pickle.html)? Pickle 🥒 é um módulo Python que serializa e desserializa uma estrutura de objeto Python. Quando você 'pickle' um modelo, você serializa ou achata sua estrutura para uso na web. Tenha cuidado: pickle não é intrinsecamente seguro, então tenha cuidado se solicitado a 'un-pickle' um arquivo. Um arquivo pickled tem o sufixo `.pkl`.

## Exercício - limpe seus dados

Nesta lição, você usará dados de 80.000 avistamentos de OVNIs, coletados pela [NUFORC](https://nuforc.org) (O Centro Nacional de Relato de OVNIs). Esses dados têm algumas descrições interessantes de avistamentos de OVNIs, por exemplo:

- **Descrição de exemplo longa.** "Um homem emerge de um feixe de luz que brilha em um campo gramado à noite e ele corre em direção ao estacionamento da Texas Instruments".
- **Descrição de exemplo curta.** "as luzes nos perseguiram".

A planilha [ufos.csv](../../../../3-Web-App/1-Web-App/data/ufos.csv) inclui colunas sobre o `city`, `state` e `country` onde o avistamento ocorreu, o `shape` do objeto e seus `latitude` e `longitude`.

No [notebook](../../../../3-Web-App/1-Web-App/notebook.ipynb) em branco incluído nesta lição:

1. importe `pandas`, `matplotlib` e `numpy` como você fez em lições anteriores e importe a planilha ufos. Você pode dar uma olhada em um conjunto de dados de exemplo:

    ```python
    import pandas as pd
    import numpy as np
    
    ufos = pd.read_csv('./data/ufos.csv')
    ufos.head()
    ```

1. Converta os dados de ufos para um pequeno dataframe com títulos novos. Verifique os valores únicos no campo `Country`.

    ```python
    ufos = pd.DataFrame({'Seconds': ufos['duration (seconds)'], 'Country': ufos['country'],'Latitude': ufos['latitude'],'Longitude': ufos['longitude']})
    
    ufos.Country.unique()
    ```

1. Agora, você pode reduzir a quantidade de dados com os quais precisamos lidar, excluindo quaisquer valores nulos e apenas importando avistamentos entre 1-60 segundos:

    ```python
    ufos.dropna(inplace=True)
    
    ufos = ufos[(ufos['Seconds'] >= 1) & (ufos['Seconds'] <= 60)]
    
    ufos.info()
    ```

1. Importe a biblioteca `LabelEncoder` do Scikit-learn para converter os valores de texto dos países em números:

    ✅ LabelEncoder codifica dados alfabeticamente

    ```python
    from sklearn.preprocessing import LabelEncoder
    
    ufos['Country'] = LabelEncoder().fit_transform(ufos['Country'])
    
    ufos.head()
    ```

    Seus dados devem parecer com isso:

    ```output
    	Seconds	Country	Latitude	Longitude
    2	20.0	3		53.200000	-2.916667
    3	20.0	4		28.978333	-96.645833
    14	30.0	4		35.823889	-80.253611
    23	60.0	4		45.582778	-122.352222
    24	3.0		3		51.783333	-0.783333
    ```

## Exercício - construa seu modelo

Agora você pode se preparar para treinar um modelo dividindo os dados em grupos de treinamento e teste.

1. Selecione os três recursos que você deseja treinar como seu vetor X, e o vetor y será `Country`. You want to be able to input `Seconds`, `Latitude` and `Longitude` e obtenha um id de país para retornar.

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

A precisão não é ruim **(cerca de 95%)**, não surpreendentemente, já que `Country` and `Latitude/Longitude` correlate.

The model you created isn't very revolutionary as you should be able to infer a `Country` from its `Latitude` and `Longitude`, mas é um bom exercício tentar treinar a partir de dados brutos que você limpou, exportou e, em seguida, usar este modelo em um app web.

## Exercício - 'pickle' seu modelo

Agora, é hora de _pickle_ seu modelo! Você pode fazer isso em algumas linhas de código. Uma vez que está _pickled_, carregue seu modelo pickled e teste-o contra um array de dados de exemplo contendo valores para segundos, latitude e longitude,

```python
import pickle
model_filename = 'ufo-model.pkl'
pickle.dump(model, open(model_filename,'wb'))

model = pickle.load(open('ufo-model.pkl','rb'))
print(model.predict([[50,44,-12]]))
```

O modelo retorna **'3'**, que é o código do país para o Reino Unido. Uau! 👽

## Exercício - construa um app Flask

Agora você pode construir um app Flask para chamar seu modelo e retornar resultados semelhantes, mas de uma maneira visualmente mais agradável.

1. Comece criando uma pasta chamada **web-app** ao lado do arquivo _notebook.ipynb_ onde seu arquivo _ufo-model.pkl_ reside.

1. Dentro dessa pasta, crie mais três pastas: **static**, com uma pasta **css** dentro dela, e **templates**. Você deve agora ter os seguintes arquivos e diretórios:

    ```output
    web-app/
      static/
        css/
      templates/
    notebook.ipynb
    ufo-model.pkl
    ```

    ✅ Consulte a pasta de soluções para uma visão do app finalizado

1. O primeiro arquivo a ser criado na pasta _web-app_ é o arquivo **requirements.txt**. Como o _package.json_ em um app JavaScript, este arquivo lista as dependências exigidas pelo app. Em **requirements.txt**, adicione as linhas:

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

1. No seu terminal, digite `pip install`, para instalar as bibliotecas listadas em _requirements.txt_:

    ```bash
    pip install -r requirements.txt
    ```

1. Agora, você está pronto para criar mais três arquivos para finalizar o app:

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

1. Em seguida, construa o arquivo _index.html_:

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

    Dê uma olhada na templateção neste arquivo. Note a sintaxe 'bigode' ao redor das variáveis que serão fornecidas pelo app, como o texto da previsão: `{{}}`. There's also a form that posts a prediction to the `/predict` route.

    Finally, you're ready to build the python file that drives the consumption of the model and the display of predictions:

1. In `app.py` adicione:

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

    > 💡 Dica: quando você adiciona [`debug=True`](https://www.askpython.com/python-modules/flask/flask-debug-mode) while running the web app using Flask, any changes you make to your application will be reflected immediately without the need to restart the server. Beware! Don't enable this mode in a production app.

If you run `python app.py` or `python3 app.py` - your web server starts up, locally, and you can fill out a short form to get an answer to your burning question about where UFOs have been sighted!

Before doing that, take a look at the parts of `app.py`:

1. First, dependencies are loaded and the app starts.
1. Then, the model is imported.
1. Then, index.html is rendered on the home route.

On the `/predict` route, several things happen when the form is posted:

1. The form variables are gathered and converted to a numpy array. They are then sent to the model and a prediction is returned.
2. The Countries that we want displayed are re-rendered as readable text from their predicted country code, and that value is sent back to index.html to be rendered in the template.

Using a model this way, with Flask and a pickled model, is relatively straightforward. The hardest thing is to understand what shape the data is that must be sent to the model to get a prediction. That all depends on how the model was trained. This one has three data points to be input in order to get a prediction.

In a professional setting, you can see how good communication is necessary between the folks who train the model and those who consume it in a web or mobile app. In our case, it's only one person, you!

---

## 🚀 Challenge

Instead of working in a notebook and importing the model to the Flask app, you could train the model right within the Flask app! Try converting your Python code in the notebook, perhaps after your data is cleaned, to train the model from within the app on a route called `train`. Quais são os prós e contras de seguir esse método?

## [Quiz pós-aula](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/18/)

## Revisão & Autoestudo

Existem muitas maneiras de construir um app web para consumir modelos de ML. Faça uma lista das maneiras que você poderia usar JavaScript ou Python para construir um app web para aproveitar o aprendizado de máquina. Considere a arquitetura: o modelo deve permanecer no app ou viver na nuvem? Se for o último, como você acessaria? Desenhe um modelo arquitetônico para uma solução web de ML aplicada.

## Tarefa

[Experimente um modelo diferente](assignment.md)

**Aviso Legal**:  
Este documento foi traduzido utilizando serviços de tradução automática baseados em IA. Embora nos esforcemos pela precisão, esteja ciente de que traduções automatizadas podem conter erros ou imprecisões. O documento original em seu idioma nativo deve ser considerado a fonte autoritária. Para informações críticas, recomenda-se a tradução profissional feita por humanos. Não nos responsabilizamos por quaisquer mal-entendidos ou interpretações errôneas decorrentes do uso desta tradução.