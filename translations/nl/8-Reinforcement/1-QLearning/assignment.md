<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "68394b2102d3503882e5e914bd0ff5c1",
  "translation_date": "2025-09-05T20:16:52+00:00",
  "source_file": "8-Reinforcement/1-QLearning/assignment.md",
  "language_code": "nl"
}
-->
# Een Meer Realistische Wereld

In onze situatie kon Peter zich bijna zonder moe te worden of honger te krijgen verplaatsen. In een meer realistische wereld moet hij af en toe gaan zitten om uit te rusten en zichzelf voeden. Laten we onze wereld realistischer maken door de volgende regels toe te passen:

1. Door van de ene plaats naar de andere te bewegen, verliest Peter **energie** en krijgt hij **vermoeidheid**.
2. Peter kan meer energie krijgen door appels te eten.
3. Peter kan vermoeidheid kwijtraken door uit te rusten onder de boom of op het gras (d.w.z. door naar een bordlocatie met een boom of gras - groen veld - te lopen).
4. Peter moet de wolf vinden en doden.
5. Om de wolf te doden, moet Peter bepaalde niveaus van energie en vermoeidheid hebben, anders verliest hij het gevecht.

## Instructies

Gebruik de originele [notebook.ipynb](../../../../8-Reinforcement/1-QLearning/notebook.ipynb) notebook als uitgangspunt voor je oplossing.

Pas de beloningsfunctie hierboven aan volgens de regels van het spel, voer het reinforcement learning-algoritme uit om de beste strategie te leren om het spel te winnen, en vergelijk de resultaten van willekeurige wandelingen met je algoritme in termen van het aantal gewonnen en verloren spellen.

> **Note**: In je nieuwe wereld is de toestand complexer en omvat deze, naast de menselijke positie, ook vermoeidheids- en energieniveaus. Je kunt ervoor kiezen om de toestand te representeren als een tuple (Board,energie,vermoeidheid), of een klasse voor de toestand te definiëren (je kunt deze ook afleiden van `Board`), of zelfs de originele `Board`-klasse aanpassen in [rlboard.py](../../../../8-Reinforcement/1-QLearning/rlboard.py).

In je oplossing, zorg ervoor dat je de code die verantwoordelijk is voor de strategie van willekeurige wandelingen behoudt, en vergelijk de resultaten van je algoritme met willekeurige wandelingen aan het einde.

> **Note**: Je moet mogelijk hyperparameters aanpassen om het te laten werken, vooral het aantal epochs. Omdat het succes van het spel (het vechten tegen de wolf) een zeldzame gebeurtenis is, kun je een veel langere trainingstijd verwachten.

## Rubric

| Criteria | Uitmuntend                                                                                                                                                                                             | Voldoende                                                                                                                                                                                | Verbetering Nodig                                                                                                                          |
| -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
|          | Een notebook wordt gepresenteerd met de definitie van nieuwe wereldregels, Q-Learning-algoritme en enkele tekstuele uitleg. Q-Learning is in staat om de resultaten aanzienlijk te verbeteren in vergelijking met willekeurige wandelingen. | Notebook wordt gepresenteerd, Q-Learning is geïmplementeerd en verbetert de resultaten in vergelijking met willekeurige wandelingen, maar niet significant; of notebook is slecht gedocumenteerd en code is niet goed gestructureerd | Er wordt een poging gedaan om de regels van de wereld opnieuw te definiëren, maar het Q-Learning-algoritme werkt niet, of de beloningsfunctie is niet volledig gedefinieerd |

---

**Disclaimer**:  
Dit document is vertaald met behulp van de AI-vertalingsservice [Co-op Translator](https://github.com/Azure/co-op-translator). Hoewel we ons best doen om nauwkeurigheid te garanderen, dient u zich ervan bewust te zijn dat geautomatiseerde vertalingen fouten of onnauwkeurigheden kunnen bevatten. Het originele document in zijn oorspronkelijke taal moet worden beschouwd als de gezaghebbende bron. Voor cruciale informatie wordt professionele menselijke vertaling aanbevolen. Wij zijn niet aansprakelijk voor eventuele misverstanden of verkeerde interpretaties die voortvloeien uit het gebruik van deze vertaling.