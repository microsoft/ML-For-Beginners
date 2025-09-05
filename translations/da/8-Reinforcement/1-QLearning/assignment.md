<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "68394b2102d3503882e5e914bd0ff5c1",
  "translation_date": "2025-09-05T01:12:36+00:00",
  "source_file": "8-Reinforcement/1-QLearning/assignment.md",
  "language_code": "da"
}
-->
# En Mere Realistisk Verden

I vores situation kunne Peter bevæge sig rundt næsten uden at blive træt eller sulten. I en mere realistisk verden skal han sætte sig ned og hvile fra tid til anden og også sørge for at spise. Lad os gøre vores verden mere realistisk ved at implementere følgende regler:

1. Ved at bevæge sig fra et sted til et andet mister Peter **energi** og opbygger noget **træthed**.
2. Peter kan få mere energi ved at spise æbler.
3. Peter kan slippe af med træthed ved at hvile under træet eller på græsset (dvs. gå ind på en brætposition med et træ eller græs - grøn mark).
4. Peter skal finde og dræbe ulven.
5. For at dræbe ulven skal Peter have visse niveauer af energi og træthed, ellers taber han kampen.

## Instruktioner

Brug den originale [notebook.ipynb](../../../../8-Reinforcement/1-QLearning/notebook.ipynb) som udgangspunkt for din løsning.

Modificer belønningsfunktionen ovenfor i henhold til spillets regler, kør forstærkningslæringsalgoritmen for at lære den bedste strategi for at vinde spillet, og sammenlign resultaterne af tilfældig gang med din algoritme med hensyn til antal vundne og tabte spil.

> **Note**: I din nye verden er tilstanden mere kompleks og inkluderer, udover menneskets position, også trætheds- og energiniveauer. Du kan vælge at repræsentere tilstanden som en tuple (Bræt, energi, træthed), eller definere en klasse for tilstanden (du kan også vælge at aflede den fra `Board`), eller endda modificere den originale `Board`-klasse inde i [rlboard.py](../../../../8-Reinforcement/1-QLearning/rlboard.py).

I din løsning skal du beholde koden, der er ansvarlig for strategien med tilfældig gang, og sammenligne resultaterne af din algoritme med tilfældig gang til sidst.

> **Note**: Du kan være nødt til at justere hyperparametre for at få det til at fungere, især antallet af epoker. Fordi succesen i spillet (kampen mod ulven) er en sjælden begivenhed, kan du forvente en meget længere træningstid.

## Vurderingskriterier

| Kriterier | Fremragende                                                                                                                                                                                             | Tilstrækkelig                                                                                                                                                                                | Kræver Forbedring                                                                                                                          |
| --------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
|           | En notebook præsenteres med definitionen af de nye verdensregler, Q-Learning-algoritmen og nogle tekstforklaringer. Q-Learning er i stand til markant at forbedre resultaterne sammenlignet med tilfældig gang. | Notebook præsenteres, Q-Learning er implementeret og forbedrer resultaterne sammenlignet med tilfældig gang, men ikke markant; eller notebook er dårligt dokumenteret, og koden er ikke velstruktureret | Der er gjort nogle forsøg på at omdefinere verdens regler, men Q-Learning-algoritmen fungerer ikke, eller belønningsfunktionen er ikke fuldt defineret |

---

**Ansvarsfraskrivelse**:  
Dette dokument er blevet oversat ved hjælp af AI-oversættelsestjenesten [Co-op Translator](https://github.com/Azure/co-op-translator). Selvom vi bestræber os på at sikre nøjagtighed, skal du være opmærksom på, at automatiserede oversættelser kan indeholde fejl eller unøjagtigheder. Det originale dokument på dets oprindelige sprog bør betragtes som den autoritative kilde. For kritisk information anbefales professionel menneskelig oversættelse. Vi er ikke ansvarlige for eventuelle misforståelser eller fejltolkninger, der måtte opstå som følge af brugen af denne oversættelse.