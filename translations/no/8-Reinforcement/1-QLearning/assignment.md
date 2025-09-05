<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "68394b2102d3503882e5e914bd0ff5c1",
  "translation_date": "2025-09-05T22:07:28+00:00",
  "source_file": "8-Reinforcement/1-QLearning/assignment.md",
  "language_code": "no"
}
-->
# En Mer Realistisk Verden

I vår situasjon kunne Peter bevege seg rundt nesten uten å bli sliten eller sulten. I en mer realistisk verden må han sette seg ned og hvile fra tid til annen, og også spise for å holde seg i live. La oss gjøre vår verden mer realistisk ved å implementere følgende regler:

1. Ved å bevege seg fra ett sted til et annet, mister Peter **energi** og får noe **utmattelse**.
2. Peter kan få mer energi ved å spise epler.
3. Peter kan kvitte seg med utmattelse ved å hvile under et tre eller på gresset (dvs. gå til en plass på brettet med et tre eller gress - grønt felt).
4. Peter må finne og drepe ulven.
5. For å drepe ulven må Peter ha visse nivåer av energi og utmattelse, ellers taper han kampen.

## Instruksjoner

Bruk den originale [notebook.ipynb](../../../../8-Reinforcement/1-QLearning/notebook.ipynb)-notatboken som utgangspunkt for løsningen din.

Modifiser belønningsfunksjonen ovenfor i henhold til spillreglene, kjør forsterkningslæringsalgoritmen for å lære den beste strategien for å vinne spillet, og sammenlign resultatene av tilfeldig vandring med algoritmen din når det gjelder antall spill vunnet og tapt.

> **Note**: I din nye verden er tilstanden mer kompleks, og inkluderer i tillegg til menneskets posisjon også nivåer av utmattelse og energi. Du kan velge å representere tilstanden som en tuple (Brett, energi, utmattelse), eller definere en klasse for tilstanden (du kan også velge å avlede den fra `Board`), eller til og med modifisere den originale `Board`-klassen i [rlboard.py](../../../../8-Reinforcement/1-QLearning/rlboard.py).

I løsningen din, vennligst behold koden som er ansvarlig for strategien med tilfeldig vandring, og sammenlign resultatene av algoritmen din med tilfeldig vandring til slutt.

> **Note**: Du kan trenge å justere hyperparametere for å få det til å fungere, spesielt antall epoker. Fordi suksessen i spillet (å bekjempe ulven) er en sjelden hendelse, kan du forvente mye lengre treningstid.

## Vurderingskriterier

| Kriterier | Fremragende                                                                                                                                                                                             | Tilfredsstillende                                                                                                                                                                                | Trenger Forbedring                                                                                                                          |
| --------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
|           | En notatbok presenteres med definisjonen av nye verdensregler, Q-Learning-algoritme og noen tekstlige forklaringer. Q-Learning er i stand til å forbedre resultatene betydelig sammenlignet med tilfeldig vandring. | Notatbok presenteres, Q-Learning er implementert og forbedrer resultatene sammenlignet med tilfeldig vandring, men ikke betydelig; eller notatboken er dårlig dokumentert og koden er ikke godt strukturert | Noen forsøk på å redefinere verdensreglene er gjort, men Q-Learning-algoritmen fungerer ikke, eller belønningsfunksjonen er ikke fullt definert |

---

**Ansvarsfraskrivelse**:  
Dette dokumentet er oversatt ved hjelp av AI-oversettelsestjenesten [Co-op Translator](https://github.com/Azure/co-op-translator). Selv om vi streber etter nøyaktighet, vær oppmerksom på at automatiske oversettelser kan inneholde feil eller unøyaktigheter. Det originale dokumentet på sitt opprinnelige språk bør anses som den autoritative kilden. For kritisk informasjon anbefales profesjonell menneskelig oversettelse. Vi er ikke ansvarlige for misforståelser eller feiltolkninger som oppstår ved bruk av denne oversettelsen.