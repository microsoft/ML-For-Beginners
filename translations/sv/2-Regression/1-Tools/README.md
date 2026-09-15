# Kom igång med Python och Scikit-learn för regressionsmodeller

![Sammanfattning av regressioner i en sketchnote](../../../../translated_images/sv/ml-regression.4e4f70e3b3ed446e.webp)

> Sketchnote av [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Förquiz innan lektionen](https://ff-quizzes.netlify.app/en/ml/)

> ### [Den här lektionen finns också på R!](../../../../2-Regression/1-Tools/solution/R/lesson_1.html)

## Introduktion

I dessa fyra lektioner kommer du att upptäcka hur man bygger regressionsmodeller. Vi diskuterar vad dessa används till inom kort. Men innan du gör något, säkerställ att du har rätt verktyg på plats för att starta processen!

I denna lektion kommer du att lära dig hur du:

- Konfigurerar din dator för lokala maskininlärningsuppgifter.
- Arbetar med Jupyter Notebooks.
- Använder Scikit-learn, inklusive installation.
- Utforskar linjär regression med en praktisk övning.

## Installationer och konfigurationer

[![ML för nybörjare - Ställ in dina verktyg redo att bygga maskininlärningsmodeller](https://img.youtube.com/vi/-DfeD2k2Kj0/0.jpg)](https://youtu.be/-DfeD2k2Kj0 "ML för nybörjare - Ställ in dina verktyg redo att bygga maskininlärningsmodeller")

> 🎥 Klicka på bilden ovan för en kort video som går igenom att konfigurera din dator för ML.

1. **Installera Python**. Se till att [Python](https://www.python.org/downloads/) är installerat på din dator. Du kommer att använda Python för många data science- och maskininlärningsuppgifter. De flesta datasystem har redan en Python-installation. Det finns också användbara [Python Coding Packs](https://code.visualstudio.com/learn/educators/installers?WT.mc_id=academic-77952-leestott) tillgängliga för att förenkla installationen för vissa användare.

   Vissa användningar av Python kräver dock en version av mjukvaran medan andra kräver en annan version. Av den anledningen är det bra att arbeta inom en [virtuell miljö](https://docs.python.org/3/library/venv.html).

2. **Installera Visual Studio Code**. Kontrollera att du har Visual Studio Code installerat på din dator. Följ dessa instruktioner för att [installera Visual Studio Code](https://code.visualstudio.com/) för en grundläggande installation. Du kommer att använda Python i Visual Studio Code i denna kurs, så du kanske vill fräscha upp dina kunskaper om hur man [konfigurerar Visual Studio Code](https://docs.microsoft.com/learn/modules/python-install-vscode?WT.mc_id=academic-77952-leestott) för Python-utveckling.

   > Bli bekväm med Python genom att gå igenom denna samling av [Lär-moduler](https://docs.microsoft.com/users/jenlooper-2911/collections/mp1pagggd5qrq7?WT.mc_id=academic-77952-leestott)
   >
   > [![Ställ in Python med Visual Studio Code](https://img.youtube.com/vi/yyQM70vi7V8/0.jpg)](https://youtu.be/yyQM70vi7V8 "Ställ in Python med Visual Studio Code")
   >
   > 🎥 Klicka på bilden ovan för en video: använda Python i VS Code.

3. **Installera Scikit-learn** genom att följa [dessa instruktioner](https://scikit-learn.org/stable/install.html). Eftersom du måste säkerställa att du använder Python 3, rekommenderas det att du använder en virtuell miljö. Observera att om du installerar detta bibliotek på en M1 Mac finns det speciella instruktioner på sidan som länkas ovan.

1. **Installera Jupyter Notebook**. Du behöver [installera Jupyter-paketet](https://pypi.org/project/jupyter/).

## Din ML-utvecklingsmiljö

Du kommer att använda **notebooks** för att utveckla din Python-kod och skapa maskininlärningsmodeller. Denna typ av fil är ett vanligt verktyg för dataforskare och de kan identifieras på deras suffix eller filändelse `.ipynb`.

Notebooks är en interaktiv miljö som tillåter utvecklaren att både koda och lägga till anteckningar och skriva dokumentation kring koden vilket är mycket hjälpsamt för experimentella eller forskningsorienterade projekt.

[![ML för nybörjare - Ställ in Jupyter Notebooks för att börja bygga regressionsmodeller](https://img.youtube.com/vi/7E-jC8FLA2E/0.jpg)](https://youtu.be/7E-jC8FLA2E "ML för nybörjare - Ställ in Jupyter Notebooks för att börja bygga regressionsmodeller")

> 🎥 Klicka på bilden ovan för en kort video som går igenom denna övning.

### Övning - arbeta med en notebook

I denna mapp hittar du filen _notebook.ipynb_.

1. Öppna _notebook.ipynb_ i Visual Studio Code.

   En Jupyter-server startar med Python 3+ igång. Du kommer att hitta områden i notebooken som kan `köras`, kodavsnitt. Du kan köra ett kodblock genom att välja ikonen som ser ut som en play-knapp.

1. Välj `md`-ikonen och lägg till lite markdown och följande text **# Välkommen till din notebook**.

   Nästa steg, lägg till lite Python-kod.

1. Skriv **print('hello notebook')** i kodblocket.
1. Välj pilen för att köra koden.

   Du bör se det utskrivna uttalandet:

    ```output
    hello notebook
    ```

![VS Code med en notebook öppen](../../../../translated_images/sv/notebook.4a3ee31f396b8832.webp)

Du kan väva in din kod med kommentarer för att själv dokumentera notebooken.

✅ Tänk en stund på hur annorlunda en webbutvecklares arbetsmiljö är jämfört med en dataforskares.

## Kom igång med Scikit-learn

Nu när Python är installerat i din lokala miljö och du är bekväm med Jupyter Notebooks, låt oss bli lika bekväma med Scikit-learn (uttalas `sci` som i `science`). Scikit-learn erbjuder ett [omfattande API](https://scikit-learn.org/stable/modules/classes.html#api-ref) som hjälper dig att utföra ML-uppgifter.

Enligt deras [webbplats](https://scikit-learn.org/stable/getting_started.html), "Scikit-learn är ett öppen källkods-maskininlärningsbibliotek som stöder övervakad och oövervakad inlärning. Det tillhandahåller också olika verktyg för modellpassning, datarensning, modellurval och utvärdering samt många andra verktyg."

I denna kurs kommer du att använda Scikit-learn och andra verktyg för att bygga maskininlärningsmodeller för att utföra vad vi kallar 'traditionella maskininlärningsuppgifter'. Vi har medvetet undvikit neurala nätverk och djupinlärning, eftersom de behandlas bättre i vår kommande kurs 'AI för nybörjare'.

Scikit-learn gör det enkelt att bygga modeller och utvärdera dem för användning. Den är främst inriktad på att använda numeriska data och innehåller flera färdiga datamängder för användning som inlärningsverktyg. Den inkluderar också förbyggda modeller för studenter att prova. Låt oss utforska processen att ladda förpackade data och använda en inbyggd estimator för att skapa din första ML-modell med Scikit-learn med viss grundläggande data.

## Övning - din första Scikit-learn notebook

> Denna tutorial är inspirerad av [exemplet för linjär regression](https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py) på Scikit-learns webbplats.


[![ML för nybörjare - Ditt första linjära regressionsprojekt i Python](https://img.youtube.com/vi/2xkXL5EUpS0/0.jpg)](https://youtu.be/2xkXL5EUpS0 "ML för nybörjare - Ditt första linjära regressionsprojekt i Python")

> 🎥 Klicka på bilden ovan för en kort video som går igenom denna övning.

I filen _notebook.ipynb_ som hör till denna lektion, rensa alla celler genom att trycka på 'papperskorgs'-ikonen.

I detta avsnitt kommer du att arbeta med en liten dataset om diabetes som är inbyggd i Scikit-learn för lärande ändamål. Föreställ dig att du ville testa en behandling för diabetiker. Maskininlärningsmodeller kan hjälpa dig att avgöra vilka patienter som skulle svara bättre på behandlingen baserat på kombinationer av variabler. Även en mycket grundläggande regressionsmodell, när den visualiseras, kan visa information om variabler som hjälper dig att organisera dina teoretiska kliniska studier.

✅ Det finns många typer av regressionsmetoder, och vilken du väljer beror på vilken fråga du vill besvara. Om du vill förutspå en sannolik längd för en person i en given ålder använder du linjär regression eftersom du söker ett **numeriskt värde**. Om du är intresserad av att upptäcka om en typ av kök ska klassas som veganskt eller inte, söker du en **kategori-klassificering**, så du skulle använda logistisk regression. Du kommer att lära dig mer om logistisk regression senare. Fundera lite på vilka frågor du kan ställa om data och vilken av dessa metoder som skulle vara lämpligare.

Låt oss börja med denna uppgift.

### Importera bibliotek

För denna uppgift kommer vi att importera några bibliotek:

- **matplotlib**. Det är ett användbart [grafverktyg](https://matplotlib.org/) som vi kommer att använda för att skapa ett linjediagram.
- **numpy**. [numpy](https://numpy.org/doc/stable/user/whatisnumpy.html) är ett användbart bibliotek för att hantera numeriska data i Python.
- **sklearn**. Det här är [Scikit-learn](https://scikit-learn.org/stable/user_guide.html)-biblioteket.

Importera några bibliotek för att underlätta dina uppgifter.

1. Lägg till imports genom att skriva följande kod:

   ```python
   import matplotlib.pyplot as plt
   import numpy as np
   from sklearn import datasets, linear_model, model_selection
   ```

   Ovan importerar du `matplotlib`, `numpy` och du importerar `datasets`, `linear_model` och `model_selection` från `sklearn`. `model_selection` används för att dela upp data i tränings- och testuppsättningar.

### Diabetes-datasetet

Det inbyggda [diabetes-datasetet](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) innehåller 442 datapunkter om diabetes med 10 egenskapsvariabler, varav några är:

- age: ålder i år
- bmi: kroppsmassindex
- bp: genomsnittligt blodtryck
- s1 tc: T-celler (en typ av vita blodkroppar)

✅ Detta dataset inkluderar begreppet 'sex' som en egenskapsvariabel som är viktig för forskning kring diabetes. Många medicinska dataset använder denna typ av binär klassifikation. Fundera lite på hur kategoriseringar som denna kan utesluta vissa delar av en befolkning från behandlingar.

Ladda nu upp X- och y-data.

> 🎓 Kom ihåg, detta är övervakad inlärning och vi behöver ett namngivet mål 'y'.

I en ny kodcell, ladda diabetes-datasetet genom att anropa `load_diabetes()`. Argumentet `return_X_y=True` signalerar att `X` kommer att vara en datamatris och `y` kommer att vara regressionsmålet.

1. Lägg till några print-kommandon för att visa formen på datamatrisen och dess första element:

    ```python
    X, y = datasets.load_diabetes(return_X_y=True)
    print(X.shape)
    print(X[0])
    ```

    Det du får tillbaka som svar är en tuple. Vad du gör är att tilldela de två första värdena i tuplen till `X` respektive `y`. Läs mer [om tupler](https://wikipedia.org/wiki/Tuple).

    Du kan se att denna data har 442 objekt formade i arrayer med 10 element:

    ```text
    (442, 10)
    [ 0.03807591  0.05068012  0.06169621  0.02187235 -0.0442235  -0.03482076
    -0.04340085 -0.00259226  0.01990842 -0.01764613]
    ```

    ✅ Fundera lite på relationen mellan data och regressionsmålet. Linjär regression förutspår relationer mellan egenskapen X och målet y. Kan du hitta [målet](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) för diabetes-datasetet i dokumentationen? Vad demonstrerar detta dataset, givet målet?

2. Välj sedan en del av detta dataset att plotta genom att välja den 3:e kolumnen i datasetet. Du kan göra detta genom att använda `:` operatorn för att välja alla rader, och sedan välja den 3:e kolumnen med index (2). Du kan också omforma datat till en 2D-array – som krävs för plotten – genom att använda `reshape(rader, kolumner)`. Om en av parametrarna är -1 beräknas motsvarande dimension automatiskt.

   ```python
   X = X[:, 2]
   X = X.reshape((-1,1))
   ```

   ✅ Skriv ut datat när som helst för att kontrollera dess form.

3. Nu när du har data redo för plot, kan du se om en maskin kan hjälpa till att bestämma en logisk uppdelning mellan talen i detta dataset. För detta behöver du dela upp både data (X) och målet (y) i test- och träningsuppsättningar. Scikit-learn har ett enkelt sätt att göra detta; du kan dela upp din testdata vid en given punkt.

   ```python
   X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.33)
   ```

4. Nu är du redo att träna din modell! Ladda in den linjära regressionsmodellen och träna den med dina X- och y-träningsmängder med `model.fit()`:

    ```python
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)
    ```

    ✅ `model.fit()` är en funktion som du kommer att se i många ML-bibliotek som TensorFlow

5. Skapa sedan en förutsägelse med testdatat, med funktionen `predict()`. Denna används för att rita linjen mellan datagrupperna.

    ```python
    y_pred = model.predict(X_test)
    ```

6. Nu är det dags att visa datat i ett diagram. Matplotlib är ett mycket användbart verktyg för denna uppgift. Skapa ett scatterplot av all X och y testdata, och använd förutsägelsen för att rita en linje på den mest lämpliga platsen mellan modellens datagrupperingar.

    ```python
    plt.scatter(X_test, y_test,  color='black')
    plt.plot(X_test, y_pred, color='blue', linewidth=3)
    plt.xlabel('Scaled BMIs')
    plt.ylabel('Disease Progression')
    plt.title('A Graph Plot Showing Diabetes Progression Against BMI')
    plt.show()
    ```

   ![ett scatterplot som visar datapunkter om diabetes](../../../../translated_images/sv/scatterplot.ad8b356bcbb33be6.webp)

   ✅ Fundera på vad som händer här. En rak linje går genom många små datapunkter, men vad gör den egentligen? Kan du se hur du ska kunna använda denna linje för att förutsäga var en ny, osedd datapunkt bör placeras i förhållande till plotens y-axel? Försök sätta ord på den praktiska användningen av denna modell.

Grattis, du har byggt din första linjära regressionsmodell, skapat en förutsägelse med den och visat den i ett diagram!

---
## 🚀Utmaning

Plotta en annan variabel från detta dataset. Tips: redigera denna rad: `X = X[:,2]`. Givet detta datasets mål, vad kan du upptäcka om diabetes som sjukdoms progression?
## [Quiz efter lektionen](https://ff-quizzes.netlify.app/en/ml/)

## Genomgång & Självstudier

I denna tutorial arbetade du med enkel linjär regression, snarare än univariat eller multipel linjär regression. Läs lite om skillnaderna mellan dessa metoder, eller titta på [den här videon](https://www.coursera.org/lecture/quantifying-relationships-regression-models/linear-vs-nonlinear-categorical-variables-ai2Ef)

Läs mer om begreppet regression och fundera på vilka typer av frågor som kan besvaras med denna teknik. Ta denna [handledning](https://docs.microsoft.com/learn/modules/train-evaluate-regression-models?WT.mc_id=academic-77952-leestott) för att fördjupa din förståelse.

## Uppgift

[En annan dataset](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Ansvarsfriskrivning**:
Detta dokument har översatts med hjälp av AI-översättningstjänsten [Co-op Translator](https://github.com/Azure/co-op-translator). Även om vi strävar efter noggrannhet, var vänlig notera att automatiska översättningar kan innehålla fel eller brister. Det ursprungliga dokumentet på dess modersmål bör betraktas som den auktoritativa källan. För kritisk information rekommenderas professionell mänsklig översättning. Vi ansvarar inte för några missförstånd eller feltolkningar som uppstår till följd av användningen av denna översättning.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->