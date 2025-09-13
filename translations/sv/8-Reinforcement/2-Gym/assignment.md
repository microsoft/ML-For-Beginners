<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "1f2b7441745eb52e25745423b247016b",
  "translation_date": "2025-09-05T22:10:42+00:00",
  "source_file": "8-Reinforcement/2-Gym/assignment.md",
  "language_code": "sv"
}
-->
# Träna Mountain Car

[OpenAI Gym](http://gym.openai.com) är utformat på ett sätt som gör att alla miljöer tillhandahåller samma API - dvs. samma metoder `reset`, `step` och `render`, samt samma abstraktioner för **aktionsutrymme** och **observationsutrymme**. Därför bör det vara möjligt att anpassa samma förstärkningsinlärningsalgoritmer till olika miljöer med minimala kodändringar.

## En Mountain Car-miljö

[Mountain Car-miljön](https://gym.openai.com/envs/MountainCar-v0/) innehåller en bil som sitter fast i en dal:

Målet är att ta sig ur dalen och fånga flaggan genom att vid varje steg utföra en av följande handlingar:

| Värde | Betydelse |
|---|---|
| 0 | Accelerera åt vänster |
| 1 | Ingen acceleration |
| 2 | Accelerera åt höger |

Huvudknepet med detta problem är dock att bilens motor inte är tillräckligt stark för att klättra uppför berget i ett enda försök. Därför är det enda sättet att lyckas att köra fram och tillbaka för att bygga upp fart.

Observationsutrymmet består av endast två värden:

| Nr | Observation  | Min | Max |
|-----|--------------|-----|-----|
|  0  | Bilens position | -1.2| 0.6 |
|  1  | Bilens hastighet | -0.07 | 0.07 |

Belöningssystemet för Mountain Car är ganska knepigt:

 * En belöning på 0 ges om agenten når flaggan (position = 0.5) på toppen av berget.
 * En belöning på -1 ges om agentens position är mindre än 0.5.

Episoden avslutas om bilens position är mer än 0.5, eller om episodens längd överstiger 200.
## Instruktioner

Anpassa vår förstärkningsinlärningsalgoritm för att lösa Mountain Car-problemet. Börja med befintlig kod i [notebook.ipynb](../../../../8-Reinforcement/2-Gym/notebook.ipynb), ersätt med den nya miljön, ändra funktionerna för diskretisering av tillstånd, och försök få den befintliga algoritmen att träna med minimala kodändringar. Optimera resultatet genom att justera hyperparametrar.

> **Obs**: Justering av hyperparametrar kommer sannolikt att behövas för att få algoritmen att konvergera. 
## Bedömningskriterier

| Kriterier | Exemplariskt | Tillräckligt | Behöver förbättras |
| -------- | --------- | -------- | ----------------- |
|          | Q-Learning-algoritmen har framgångsrikt anpassats från CartPole-exemplet med minimala kodändringar och kan lösa problemet med att fånga flaggan på under 200 steg. | En ny Q-Learning-algoritm har hämtats från internet, men är väl dokumenterad; eller befintlig algoritm har anpassats men når inte önskade resultat. | Studenten har inte lyckats anpassa någon algoritm framgångsrikt, men har gjort betydande framsteg mot en lösning (implementerat tillståndsdiskretisering, Q-Tabell-datastruktur, etc.) |

---

**Ansvarsfriskrivning**:  
Detta dokument har översatts med hjälp av AI-översättningstjänsten [Co-op Translator](https://github.com/Azure/co-op-translator). Även om vi strävar efter noggrannhet, vänligen notera att automatiska översättningar kan innehålla fel eller felaktigheter. Det ursprungliga dokumentet på dess originalspråk bör betraktas som den auktoritativa källan. För kritisk information rekommenderas professionell mänsklig översättning. Vi ansvarar inte för eventuella missförstånd eller feltolkningar som uppstår vid användning av denna översättning.