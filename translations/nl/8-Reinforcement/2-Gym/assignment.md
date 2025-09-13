<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "1f2b7441745eb52e25745423b247016b",
  "translation_date": "2025-09-05T20:22:44+00:00",
  "source_file": "8-Reinforcement/2-Gym/assignment.md",
  "language_code": "nl"
}
-->
# Train Mountain Car

[OpenAI Gym](http://gym.openai.com) is zo ontworpen dat alle omgevingen dezelfde API bieden - namelijk dezelfde methoden `reset`, `step` en `render`, en dezelfde abstracties van **actie ruimte** en **observatie ruimte**. Hierdoor zou het mogelijk moeten zijn om dezelfde reinforcement learning-algoritmes aan te passen aan verschillende omgevingen met minimale codewijzigingen.

## Een Mountain Car-omgeving

[Mountain Car-omgeving](https://gym.openai.com/envs/MountainCar-v0/) bevat een auto die vastzit in een vallei:

Het doel is om uit de vallei te komen en de vlag te bereiken door bij elke stap een van de volgende acties uit te voeren:

| Waarde | Betekenis |
|---|---|
| 0 | Versnellen naar links |
| 1 | Niet versnellen |
| 2 | Versnellen naar rechts |

De belangrijkste uitdaging van dit probleem is echter dat de motor van de auto niet sterk genoeg is om de berg in één keer te beklimmen. Daarom is de enige manier om te slagen heen en weer rijden om momentum op te bouwen.

De observatieruimte bestaat slechts uit twee waarden:

| Nr | Observatie  | Min | Max |
|-----|--------------|-----|-----|
|  0  | Positie van de auto | -1.2| 0.6 |
|  1  | Snelheid van de auto | -0.07 | 0.07 |

Het beloningssysteem voor de mountain car is vrij ingewikkeld:

 * Een beloning van 0 wordt toegekend als de agent de vlag heeft bereikt (positie = 0.5) bovenop de berg.
 * Een beloning van -1 wordt toegekend als de positie van de agent minder dan 0.5 is.

De episode eindigt als de positie van de auto meer dan 0.5 is, of als de lengte van de episode groter is dan 200.
## Instructies

Pas ons reinforcement learning-algoritme aan om het mountain car-probleem op te lossen. Begin met de bestaande [notebook.ipynb](../../../../8-Reinforcement/2-Gym/notebook.ipynb) code, vervang de omgeving, wijzig de functies voor staat-discretisatie, en probeer het bestaande algoritme te trainen met minimale codewijzigingen. Optimaliseer het resultaat door de hyperparameters aan te passen.

> **Let op**: Het aanpassen van hyperparameters is waarschijnlijk nodig om het algoritme te laten convergeren. 
## Rubriek

| Criteria | Uitmuntend | Voldoende | Verbetering nodig |
| -------- | --------- | -------- | ----------------- |
|          | Q-Learning-algoritme is succesvol aangepast van het CartPole-voorbeeld, met minimale codewijzigingen, en is in staat om het probleem van het bereiken van de vlag binnen 200 stappen op te lossen. | Een nieuw Q-Learning-algoritme is overgenomen van het internet, maar goed gedocumenteerd; of bestaand algoritme is aangepast, maar bereikt niet de gewenste resultaten. | Student was niet in staat om een algoritme succesvol aan te passen, maar heeft aanzienlijke stappen gezet richting een oplossing (zoals het implementeren van staat-discretisatie, Q-Table datastructuur, etc.) |

---

**Disclaimer**:  
Dit document is vertaald met behulp van de AI-vertalingsservice [Co-op Translator](https://github.com/Azure/co-op-translator). Hoewel we streven naar nauwkeurigheid, dient u zich ervan bewust te zijn dat geautomatiseerde vertalingen fouten of onnauwkeurigheden kunnen bevatten. Het originele document in zijn oorspronkelijke taal moet worden beschouwd als de gezaghebbende bron. Voor cruciale informatie wordt professionele menselijke vertaling aanbevolen. Wij zijn niet aansprakelijk voor eventuele misverstanden of verkeerde interpretaties die voortvloeien uit het gebruik van deze vertaling.