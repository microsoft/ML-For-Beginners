<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "e0b75f73e4a90d45181dc5581fe2ef5c",
  "translation_date": "2025-09-05T08:45:52+00:00",
  "source_file": "3-Web-App/1-Web-App/README.md",
  "language_code": "pt"
}
-->
# Construir uma Aplicação Web para Utilizar um Modelo de ML

Nesta lição, vais treinar um modelo de ML com um conjunto de dados fora do comum: _avistamentos de OVNIs ao longo do último século_, provenientes da base de dados do NUFORC.

Vais aprender:

- Como 'pickle' um modelo treinado
- Como usar esse modelo numa aplicação Flask

Continuaremos a usar notebooks para limpar os dados e treinar o modelo, mas podes levar o processo um passo adiante ao explorar como usar um modelo "no mundo real", por assim dizer: numa aplicação web.

Para isso, precisas de construir uma aplicação web utilizando Flask.

## [Questionário pré-aula](https://ff-quizzes.netlify.app/en/ml/)

## Construir uma aplicação

Existem várias formas de construir aplicações web para consumir modelos de machine learning. A arquitetura da tua aplicação web pode influenciar a forma como o modelo é treinado. Imagina que estás a trabalhar numa empresa onde o grupo de ciência de dados treinou um modelo que querem que utilizes numa aplicação.

### Considerações

Há muitas perguntas que precisas de fazer:

- **É uma aplicação web ou uma aplicação móvel?** Se estás a construir uma aplicação móvel ou precisas de usar o modelo num contexto de IoT, podes usar [TensorFlow Lite](https://www.tensorflow.org/lite/) e utilizar o modelo numa aplicação Android ou iOS.
- **Onde o modelo vai residir?** Na nuvem ou localmente?
- **Suporte offline.** A aplicação precisa de funcionar offline?
- **Que tecnologia foi usada para treinar o modelo?** A tecnologia escolhida pode influenciar as ferramentas que precisas de usar.
    - **Usando TensorFlow.** Se estás a treinar um modelo com TensorFlow, por exemplo, esse ecossistema permite converter um modelo TensorFlow para uso numa aplicação web utilizando [TensorFlow.js](https://www.tensorflow.org/js/).
    - **Usando PyTorch.** Se estás a construir um modelo com uma biblioteca como [PyTorch](https://pytorch.org/), tens a opção de exportá-lo no formato [ONNX](https://onnx.ai/) (Open Neural Network Exchange) para uso em aplicações web JavaScript que podem utilizar o [Onnx Runtime](https://www.onnxruntime.ai/). Esta opção será explorada numa lição futura para um modelo treinado com Scikit-learn.
    - **Usando Lobe.ai ou Azure Custom Vision.** Se estás a usar um sistema ML SaaS (Software como Serviço) como [Lobe.ai](https://lobe.ai/) ou [Azure Custom Vision](https://azure.microsoft.com/services/cognitive-services/custom-vision-service/?WT.mc_id=academic-77952-leestott) para treinar um modelo, este tipo de software fornece formas de exportar o modelo para várias plataformas, incluindo construir uma API personalizada para ser consultada na nuvem pela tua aplicação online.

Também tens a oportunidade de construir uma aplicação web Flask completa que seria capaz de treinar o modelo diretamente no navegador. Isso também pode ser feito utilizando TensorFlow.js num contexto JavaScript.

Para os nossos propósitos, como temos trabalhado com notebooks baseados em Python, vamos explorar os passos necessários para exportar um modelo treinado de um notebook para um formato legível por uma aplicação web construída em Python.

## Ferramenta

Para esta tarefa, precisas de duas ferramentas: Flask e Pickle, ambas executadas em Python.

✅ O que é [Flask](https://palletsprojects.com/p/flask/)? Definido como um 'micro-framework' pelos seus criadores, Flask fornece as funcionalidades básicas de frameworks web utilizando Python e um motor de templates para construir páginas web. Dá uma olhada neste [módulo de aprendizagem](https://docs.microsoft.com/learn/modules/python-flask-build-ai-web-app?WT.mc_id=academic-77952-leestott) para praticar a construção com Flask.

✅ O que é [Pickle](https://docs.python.org/3/library/pickle.html)? Pickle 🥒 é um módulo Python que serializa e desserializa uma estrutura de objetos Python. Quando 'pickle' um modelo, estás a serializar ou achatar a sua estrutura para uso na web. Atenção: pickle não é intrinsecamente seguro, por isso tem cuidado se fores solicitado a 'des-picklar' um ficheiro. Um ficheiro pickled tem o sufixo `.pkl`.

## Exercício - limpar os dados

Nesta lição vais usar dados de 80.000 avistamentos de OVNIs, recolhidos pelo [NUFORC](https://nuforc.org) (Centro Nacional de Relatórios de OVNIs). Estes dados têm descrições interessantes de avistamentos de OVNIs, por exemplo:

- **Descrição longa de exemplo.** "Um homem emerge de um feixe de luz que brilha num campo de relva à noite e corre em direção ao estacionamento da Texas Instruments".
- **Descrição curta de exemplo.** "as luzes perseguiram-nos".

A folha de cálculo [ufos.csv](../../../../3-Web-App/1-Web-App/data/ufos.csv) inclui colunas sobre a `cidade`, `estado` e `país` onde o avistamento ocorreu, a `forma` do objeto e a sua `latitude` e `longitude`.

No [notebook](../../../../3-Web-App/1-Web-App/notebook.ipynb) em branco incluído nesta lição:

1. Importa `pandas`, `matplotlib` e `numpy` como fizeste nas lições anteriores e importa a folha de cálculo de OVNIs. Podes dar uma olhada num conjunto de dados de exemplo:

    ```python
    import pandas as pd
    import numpy as np
    
    ufos = pd.read_csv('./data/ufos.csv')
    ufos.head()
    ```

1. Converte os dados de OVNIs para um pequeno dataframe com títulos novos. Verifica os valores únicos no campo `Country`.

    ```python
    ufos = pd.DataFrame({'Seconds': ufos['duration (seconds)'], 'Country': ufos['country'],'Latitude': ufos['latitude'],'Longitude': ufos['longitude']})
    
    ufos.Country.unique()
    ```

1. Agora, podes reduzir a quantidade de dados com que precisas de lidar ao eliminar valores nulos e importar apenas avistamentos entre 1-60 segundos:

    ```python
    ufos.dropna(inplace=True)
    
    ufos = ufos[(ufos['Seconds'] >= 1) & (ufos['Seconds'] <= 60)]
    
    ufos.info()
    ```

1. Importa a biblioteca `LabelEncoder` do Scikit-learn para converter os valores de texto dos países para números:

    ✅ LabelEncoder codifica os dados alfabeticamente

    ```python
    from sklearn.preprocessing import LabelEncoder
    
    ufos['Country'] = LabelEncoder().fit_transform(ufos['Country'])
    
    ufos.head()
    ```

    Os teus dados devem parecer-se com isto:

    ```output
    	Seconds	Country	Latitude	Longitude
    2	20.0	3		53.200000	-2.916667
    3	20.0	4		28.978333	-96.645833
    14	30.0	4		35.823889	-80.253611
    23	60.0	4		45.582778	-122.352222
    24	3.0		3		51.783333	-0.783333
    ```

## Exercício - construir o modelo

Agora podes preparar-te para treinar um modelo dividindo os dados em grupos de treino e teste.

1. Seleciona as três características que queres treinar como o teu vetor X, e o vetor y será o `Country`. Queres ser capaz de inserir `Seconds`, `Latitude` e `Longitude` e obter um id de país como retorno.

    ```python
    from sklearn.model_selection import train_test_split
    
    Selected_features = ['Seconds','Latitude','Longitude']
    
    X = ufos[Selected_features]
    y = ufos['Country']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    ```

1. Treina o modelo utilizando regressão logística:

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

A precisão não é má **(cerca de 95%)**, o que não é surpreendente, já que `Country` e `Latitude/Longitude` estão correlacionados.

O modelo que criaste não é muito revolucionário, pois deverias ser capaz de inferir um `Country` a partir da sua `Latitude` e `Longitude`, mas é um bom exercício para tentar treinar a partir de dados brutos que limpaste, exportaste e depois usaste este modelo numa aplicação web.

## Exercício - 'pickle' o modelo

Agora, é hora de _picklar_ o modelo! Podes fazer isso em algumas linhas de código. Uma vez _pickled_, carrega o modelo pickled e testa-o contra um array de dados de exemplo contendo valores para segundos, latitude e longitude.

```python
import pickle
model_filename = 'ufo-model.pkl'
pickle.dump(model, open(model_filename,'wb'))

model = pickle.load(open('ufo-model.pkl','rb'))
print(model.predict([[50,44,-12]]))
```

O modelo retorna **'3'**, que é o código de país para o Reino Unido. Incrível! 👽

## Exercício - construir uma aplicação Flask

Agora podes construir uma aplicação Flask para chamar o modelo e retornar resultados semelhantes, mas de uma forma mais visualmente agradável.

1. Começa por criar uma pasta chamada **web-app** ao lado do ficheiro _notebook.ipynb_ onde o teu ficheiro _ufo-model.pkl_ reside.

1. Nessa pasta, cria mais três pastas: **static**, com uma pasta **css** dentro dela, e **templates**. Deves agora ter os seguintes ficheiros e diretórios:

    ```output
    web-app/
      static/
        css/
      templates/
    notebook.ipynb
    ufo-model.pkl
    ```

    ✅ Consulta a pasta de solução para ver a aplicação finalizada

1. O primeiro ficheiro a criar na pasta _web-app_ é o ficheiro **requirements.txt**. Tal como _package.json_ numa aplicação JavaScript, este ficheiro lista as dependências necessárias para a aplicação. No **requirements.txt** adiciona as linhas:

    ```text
    scikit-learn
    pandas
    numpy
    flask
    ```

1. Agora, executa este ficheiro navegando até _web-app_:

    ```bash
    cd web-app
    ```

1. No terminal, digita `pip install` para instalar as bibliotecas listadas no _requirements.txt_:

    ```bash
    pip install -r requirements.txt
    ```

1. Agora, estás pronto para criar mais três ficheiros para finalizar a aplicação:

    1. Cria **app.py** na raiz.
    2. Cria **index.html** na pasta _templates_.
    3. Cria **styles.css** na pasta _static/css_.

1. Desenvolve o ficheiro _styles.css_ com alguns estilos:

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

1. Em seguida, desenvolve o ficheiro _index.html_:

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

    Dá uma olhada na utilização de templates neste ficheiro. Repara na sintaxe 'mustache' em torno das variáveis que serão fornecidas pela aplicação, como o texto de previsão: `{{}}`. Há também um formulário que envia uma previsão para a rota `/predict`.

    Finalmente, estás pronto para construir o ficheiro Python que conduz o consumo do modelo e a exibição das previsões:

1. No `app.py` adiciona:

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

    > 💡 Dica: quando adicionas [`debug=True`](https://www.askpython.com/python-modules/flask/flask-debug-mode) enquanto executas a aplicação web utilizando Flask, quaisquer alterações que fizeres na tua aplicação serão refletidas imediatamente sem necessidade de reiniciar o servidor. Atenção! Não habilites este modo numa aplicação em produção.

Se executares `python app.py` ou `python3 app.py` - o teu servidor web inicia localmente, e podes preencher um pequeno formulário para obter uma resposta à tua pergunta sobre onde os OVNIs foram avistados!

Antes de fazer isso, dá uma olhada nas partes do `app.py`:

1. Primeiro, as dependências são carregadas e a aplicação inicia.
1. Depois, o modelo é importado.
1. Em seguida, o index.html é renderizado na rota inicial.

Na rota `/predict`, várias coisas acontecem quando o formulário é enviado:

1. As variáveis do formulário são recolhidas e convertidas para um array numpy. Elas são então enviadas para o modelo e uma previsão é retornada.
2. Os países que queremos exibir são re-renderizados como texto legível a partir do código de país previsto, e esse valor é enviado de volta ao index.html para ser renderizado no template.

Usar um modelo desta forma, com Flask e um modelo pickled, é relativamente simples. O mais difícil é entender qual é a forma dos dados que devem ser enviados ao modelo para obter uma previsão. Isso depende de como o modelo foi treinado. Este tem três pontos de dados que devem ser inseridos para obter uma previsão.

Num ambiente profissional, podes ver como é necessária uma boa comunicação entre as pessoas que treinam o modelo e aquelas que o consomem numa aplicação web ou móvel. No nosso caso, és apenas tu!

---

## 🚀 Desafio

Em vez de trabalhar num notebook e importar o modelo para a aplicação Flask, poderias treinar o modelo diretamente dentro da aplicação Flask! Tenta converter o teu código Python no notebook, talvez depois de os dados serem limpos, para treinar o modelo dentro da aplicação numa rota chamada `train`. Quais são os prós e contras de seguir este método?

## [Questionário pós-aula](https://ff-quizzes.netlify.app/en/ml/)

## Revisão & Autoestudo

Existem muitas formas de construir uma aplicação web para consumir modelos de ML. Faz uma lista das formas como poderias usar JavaScript ou Python para construir uma aplicação web que aproveite o machine learning. Considera a arquitetura: o modelo deve permanecer na aplicação ou viver na nuvem? Se for o último caso, como o acederias? Desenha um modelo arquitetural para uma solução web aplicada de ML.

## Tarefa

[Experimenta um modelo diferente](assignment.md)

---

**Aviso Legal**:  
Este documento foi traduzido utilizando o serviço de tradução automática [Co-op Translator](https://github.com/Azure/co-op-translator). Embora nos esforcemos para garantir a precisão, esteja ciente de que traduções automáticas podem conter erros ou imprecisões. O documento original na sua língua nativa deve ser considerado a fonte oficial. Para informações críticas, recomenda-se a tradução profissional realizada por humanos. Não nos responsabilizamos por quaisquer mal-entendidos ou interpretações incorretas resultantes do uso desta tradução.