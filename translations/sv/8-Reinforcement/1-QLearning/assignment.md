<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "68394b2102d3503882e5e914bd0ff5c1",
  "translation_date": "2025-09-05T22:07:07+00:00",
  "source_file": "8-Reinforcement/1-QLearning/assignment.md",
  "language_code": "sv"
}
-->
# En Mer Realistisk Värld

I vår situation kunde Peter röra sig nästan utan att bli trött eller hungrig. I en mer realistisk värld måste han sätta sig ner och vila då och då, och även äta för att hålla sig mätt. Låt oss göra vår värld mer realistisk genom att implementera följande regler:

1. När Peter rör sig från en plats till en annan förlorar han **energi** och får **trötthet**.
2. Peter kan få mer energi genom att äta äpplen.
3. Peter kan bli av med trötthet genom att vila under trädet eller på gräset (dvs. gå till en plats på spelplanen med ett träd eller gräs - grönt fält).
4. Peter måste hitta och döda vargen.
5. För att döda vargen måste Peter ha vissa nivåer av energi och trötthet, annars förlorar han striden.

## Instruktioner

Använd den ursprungliga [notebook.ipynb](../../../../8-Reinforcement/1-QLearning/notebook.ipynb) som utgångspunkt för din lösning.

Modifiera belöningsfunktionen ovan enligt spelets regler, kör förstärkningsinlärningsalgoritmen för att lära dig den bästa strategin för att vinna spelet, och jämför resultaten av slumpmässig gång med din algoritm i termer av antal vunna och förlorade spel.

> **Note**: I din nya värld är tillståndet mer komplext och inkluderar, förutom människans position, även nivåer av trötthet och energi. Du kan välja att representera tillståndet som en tuple (Board,energy,fatigue), eller definiera en klass för tillståndet (du kan också vilja härleda den från `Board`), eller till och med modifiera den ursprungliga `Board`-klassen i [rlboard.py](../../../../8-Reinforcement/1-QLearning/rlboard.py).

I din lösning, behåll koden som ansvarar för strategin med slumpmässig gång och jämför resultaten av din algoritm med slumpmässig gång i slutet.

> **Note**: Du kan behöva justera hyperparametrar för att få det att fungera, särskilt antalet epoker. Eftersom spelets framgång (att slåss mot vargen) är en sällsynt händelse kan du förvänta dig mycket längre träningstid.

## Bedömningskriterier

| Kriterier | Exemplariskt                                                                                                                                                                                             | Tillräckligt                                                                                                                                                                             | Behöver Förbättras                                                                                                                         |
| --------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
|           | En notebook presenteras med definitionen av nya världens regler, Q-Learning-algoritmen och några textförklaringar. Q-Learning kan avsevärt förbättra resultaten jämfört med slumpmässig gång.             | Notebook presenteras, Q-Learning är implementerad och förbättrar resultaten jämfört med slumpmässig gång, men inte avsevärt; eller notebook är dåligt dokumenterad och koden är inte välstrukturerad | Några försök att omdefiniera världens regler görs, men Q-Learning-algoritmen fungerar inte, eller belöningsfunktionen är inte fullt definierad |

---

**Ansvarsfriskrivning**:  
Detta dokument har översatts med hjälp av AI-översättningstjänsten [Co-op Translator](https://github.com/Azure/co-op-translator). Även om vi strävar efter noggrannhet, bör det noteras att automatiserade översättningar kan innehålla fel eller brister. Det ursprungliga dokumentet på dess originalspråk bör betraktas som den auktoritativa källan. För kritisk information rekommenderas professionell mänsklig översättning. Vi ansvarar inte för eventuella missförstånd eller feltolkningar som uppstår vid användning av denna översättning.