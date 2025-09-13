<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "9d91f3af3758fdd4569fb410575995ef",
  "translation_date": "2025-09-05T21:40:53+00:00",
  "source_file": "1-Introduction/4-techniques-of-ML/README.md",
  "language_code": "sv"
}
-->
# Tekniker för maskininlärning

Processen att bygga, använda och underhålla modeller för maskininlärning och den data de använder skiljer sig avsevärt från många andra utvecklingsarbetsflöden. I denna lektion kommer vi att avmystifiera processen och beskriva de huvudsakliga tekniker du behöver känna till. Du kommer att:

- Förstå de processer som ligger till grund för maskininlärning på en övergripande nivå.
- Utforska grundläggande begrepp som "modeller", "prediktioner" och "träningsdata".

## [Quiz före föreläsningen](https://ff-quizzes.netlify.app/en/ml/)

[![ML för nybörjare - Tekniker för maskininlärning](https://img.youtube.com/vi/4NGM0U2ZSHU/0.jpg)](https://youtu.be/4NGM0U2ZSHU "ML för nybörjare - Tekniker för maskininlärning")

> 🎥 Klicka på bilden ovan för en kort video som går igenom denna lektion.

## Introduktion

På en övergripande nivå består hantverket att skapa processer för maskininlärning (ML) av flera steg:

1. **Bestäm frågan**. De flesta ML-processer börjar med att ställa en fråga som inte kan besvaras med ett enkelt villkorsprogram eller en regelbaserad motor. Dessa frågor handlar ofta om att göra prediktioner baserade på en samling data.
2. **Samla in och förbered data**. För att kunna besvara din fråga behöver du data. Kvaliteten och ibland mängden av din data kommer att avgöra hur väl du kan besvara din ursprungliga fråga. Att visualisera data är en viktig del av denna fas. Denna fas inkluderar också att dela upp data i en tränings- och testgrupp för att bygga en modell.
3. **Välj en träningsmetod**. Beroende på din fråga och datans natur behöver du välja hur du vill träna en modell för att bäst reflektera din data och göra korrekta prediktioner baserat på den. Detta är den del av din ML-process som kräver specifik expertis och ofta en betydande mängd experimenterande.
4. **Träna modellen**. Med hjälp av din träningsdata använder du olika algoritmer för att träna en modell att känna igen mönster i datan. Modellen kan använda interna vikter som kan justeras för att prioritera vissa delar av datan över andra för att bygga en bättre modell.
5. **Utvärdera modellen**. Du använder data som modellen aldrig tidigare sett (din testdata) från din insamlade uppsättning för att se hur modellen presterar.
6. **Justera parametrar**. Baserat på modellens prestanda kan du göra om processen med olika parametrar eller variabler som styr beteendet hos de algoritmer som används för att träna modellen.
7. **Prediktera**. Använd nya indata för att testa modellens noggrannhet.

## Vilken fråga ska ställas?

Datorer är särskilt skickliga på att upptäcka dolda mönster i data. Denna förmåga är mycket användbar för forskare som har frågor om ett visst område som inte enkelt kan besvaras genom att skapa en regelbaserad motor. Givet en aktuarieuppgift, till exempel, kan en dataforskare skapa handgjorda regler kring dödligheten hos rökare jämfört med icke-rökare.

När många andra variabler tas med i ekvationen kan dock en ML-modell visa sig vara mer effektiv för att förutsäga framtida dödlighetsnivåer baserat på tidigare hälsodata. Ett mer positivt exempel kan vara att göra väderprognoser för april månad på en viss plats baserat på data som inkluderar latitud, longitud, klimatförändringar, närhet till havet, jetströmmens mönster och mer.

✅ Denna [presentation](https://www2.cisl.ucar.edu/sites/default/files/2021-10/0900%20June%2024%20Haupt_0.pdf) om vädermodeller ger ett historiskt perspektiv på att använda ML i väderanalys.  

## Förberedande uppgifter

Innan du börjar bygga din modell finns det flera uppgifter du behöver slutföra. För att testa din fråga och formulera en hypotes baserad på modellens prediktioner behöver du identifiera och konfigurera flera element.

### Data

För att kunna besvara din fråga med någon form av säkerhet behöver du en tillräcklig mängd data av rätt typ. Det finns två saker du behöver göra vid denna punkt:

- **Samla in data**. Med tanke på den tidigare lektionen om rättvisa i dataanalys, samla in din data med omsorg. Var medveten om källorna till denna data, eventuella inneboende fördomar den kan ha, och dokumentera dess ursprung.
- **Förbered data**. Det finns flera steg i processen att förbereda data. Du kan behöva sammanställa data och normalisera den om den kommer från olika källor. Du kan förbättra datans kvalitet och kvantitet genom olika metoder, såsom att konvertera strängar till siffror (som vi gör i [Klustring](../../5-Clustering/1-Visualize/README.md)). Du kan också generera ny data baserat på den ursprungliga (som vi gör i [Klassificering](../../4-Classification/1-Introduction/README.md)). Du kan rensa och redigera datan (som vi gör inför [Webbapplikationslektionen](../../3-Web-App/README.md)). Slutligen kan du behöva slumpa och blanda datan, beroende på dina träningsmetoder.

✅ Efter att ha samlat in och bearbetat din data, ta en stund för att se om dess struktur tillåter dig att adressera din avsedda fråga. Det kan vara så att datan inte presterar väl i din givna uppgift, som vi upptäcker i våra [Klustringslektioner](../../5-Clustering/1-Visualize/README.md)!

### Funktioner och mål

En [funktion](https://www.datasciencecentral.com/profiles/blogs/an-introduction-to-variable-and-feature-selection) är en mätbar egenskap hos din data. I många dataset uttrycks det som en kolumnrubrik som "datum", "storlek" eller "färg". Din funktionsvariabel, vanligtvis representerad som `X` i kod, representerar indata som kommer att användas för att träna modellen.

Ett mål är det du försöker förutsäga. Målet, vanligtvis representerat som `y` i kod, representerar svaret på den fråga du försöker ställa till din data: i december, vilken **färg** på pumpor kommer att vara billigast? I San Francisco, vilka områden kommer att ha det bästa fastighets**priset**? Ibland kallas målet också för etikettattribut.

### Välja din funktionsvariabel

🎓 **Funktionsval och funktionsutvinning** Hur vet du vilken variabel du ska välja när du bygger en modell? Du kommer förmodligen att gå igenom en process av funktionsval eller funktionsutvinning för att välja rätt variabler för den mest presterande modellen. De är dock inte samma sak: "Funktionsutvinning skapar nya funktioner från funktioner av de ursprungliga funktionerna, medan funktionsval returnerar en delmängd av funktionerna." ([källa](https://wikipedia.org/wiki/Feature_selection))

### Visualisera din data

En viktig aspekt av dataforskarens verktygslåda är förmågan att visualisera data med hjälp av flera utmärkta bibliotek som Seaborn eller MatPlotLib. Att representera din data visuellt kan hjälpa dig att upptäcka dolda korrelationer som du kan utnyttja. Dina visualiseringar kan också hjälpa dig att upptäcka fördomar eller obalanserad data (som vi upptäcker i [Klassificering](../../4-Classification/2-Classifiers-1/README.md)).

### Dela upp din dataset

Innan träning behöver du dela upp din dataset i två eller fler delar av olika storlek som fortfarande representerar datan väl.

- **Träning**. Denna del av datasetet används för att träna din modell. Denna uppsättning utgör majoriteten av den ursprungliga datasetet.
- **Testning**. En testdataset är en oberoende grupp av data, ofta hämtad från den ursprungliga datan, som du använder för att bekräfta prestandan hos den byggda modellen.
- **Validering**. En valideringsuppsättning är en mindre oberoende grupp av exempel som du använder för att justera modellens hyperparametrar eller arkitektur för att förbättra modellen. Beroende på storleken på din data och frågan du ställer kanske du inte behöver bygga denna tredje uppsättning (som vi noterar i [Tidsserieprognoser](../../7-TimeSeries/1-Introduction/README.md)).

## Bygga en modell

Med hjälp av din träningsdata är ditt mål att bygga en modell, eller en statistisk representation av din data, med hjälp av olika algoritmer för att **träna** den. Att träna en modell exponerar den för data och låter den göra antaganden om uppfattade mönster den upptäcker, validerar och accepterar eller förkastar.

### Bestäm en träningsmetod

Beroende på din fråga och datans natur kommer du att välja en metod för att träna den. Genom att gå igenom [Scikit-learns dokumentation](https://scikit-learn.org/stable/user_guide.html) - som vi använder i denna kurs - kan du utforska många sätt att träna en modell. Beroende på din erfarenhet kan du behöva prova flera olika metoder för att bygga den bästa modellen. Du kommer sannolikt att gå igenom en process där dataforskare utvärderar modellens prestanda genom att mata in data den inte har sett tidigare, kontrollera noggrannhet, fördomar och andra kvalitetsförsämrande problem, och välja den mest lämpliga träningsmetoden för den aktuella uppgiften.

### Träna en modell

Med din träningsdata redo är du redo att "anpassa" den för att skapa en modell. Du kommer att märka att i många ML-bibliotek hittar du koden 'model.fit' - det är vid denna tidpunkt som du skickar in din funktionsvariabel som en array av värden (vanligtvis 'X') och en målvariabel (vanligtvis 'y').

### Utvärdera modellen

När träningsprocessen är klar (det kan ta många iterationer, eller "epoker", att träna en stor modell) kommer du att kunna utvärdera modellens kvalitet genom att använda testdata för att bedöma dess prestanda. Denna data är en delmängd av den ursprungliga datan som modellen inte tidigare har analyserat. Du kan skriva ut en tabell med mätvärden om modellens kvalitet.

🎓 **Modellanpassning**

I maskininlärningens kontext hänvisar modellanpassning till modellens noggrannhet när den försöker analysera data som den inte är bekant med.

🎓 **Underanpassning** och **överanpassning** är vanliga problem som försämrar modellens kvalitet, eftersom modellen anpassar sig antingen inte tillräckligt bra eller för bra. Detta gör att modellen gör prediktioner som antingen är för nära eller för löst kopplade till dess träningsdata. En överanpassad modell förutsäger träningsdata för bra eftersom den har lärt sig datans detaljer och brus för bra. En underanpassad modell är inte noggrann eftersom den varken kan analysera sin träningsdata eller data den inte har "sett" korrekt.

![överanpassad modell](../../../../1-Introduction/4-techniques-of-ML/images/overfitting.png)
> Infografik av [Jen Looper](https://twitter.com/jenlooper)

## Justera parametrar

När din initiala träning är klar, observera modellens kvalitet och överväg att förbättra den genom att justera dess "hyperparametrar". Läs mer om processen [i dokumentationen](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters?WT.mc_id=academic-77952-leestott).

## Prediktion

Detta är ögonblicket då du kan använda helt ny data för att testa modellens noggrannhet. I en "tillämpad" ML-miljö, där du bygger webbapplikationer för att använda modellen i produktion, kan denna process innebära att samla in användarinmatning (till exempel ett knapptryck) för att ställa in en variabel och skicka den till modellen för inferens eller utvärdering.

I dessa lektioner kommer du att upptäcka hur du använder dessa steg för att förbereda, bygga, testa, utvärdera och prediktera - alla gester av en dataforskare och mer, när du utvecklas i din resa att bli en "fullstack"-ML-ingenjör.

---

## 🚀Utmaning

Rita ett flödesschema som reflekterar stegen för en ML-praktiker. Var befinner du dig just nu i processen? Var tror du att du kommer att stöta på svårigheter? Vad verkar enkelt för dig?

## [Quiz efter föreläsningen](https://ff-quizzes.netlify.app/en/ml/)

## Granskning & Självstudier

Sök online efter intervjuer med dataforskare som diskuterar sitt dagliga arbete. Här är [en](https://www.youtube.com/watch?v=Z3IjgbbCEfs).

## Uppgift

[Intervjua en dataforskare](assignment.md)

---

**Ansvarsfriskrivning**:  
Detta dokument har översatts med hjälp av AI-översättningstjänsten [Co-op Translator](https://github.com/Azure/co-op-translator). Även om vi strävar efter noggrannhet, vänligen notera att automatiska översättningar kan innehålla fel eller felaktigheter. Det ursprungliga dokumentet på dess originalspråk bör betraktas som den auktoritativa källan. För kritisk information rekommenderas professionell mänsklig översättning. Vi ansvarar inte för eventuella missförstånd eller feltolkningar som uppstår vid användning av denna översättning.