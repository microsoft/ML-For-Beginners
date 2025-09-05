<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "1f2b7441745eb52e25745423b247016b",
  "translation_date": "2025-09-05T01:18:36+00:00",
  "source_file": "8-Reinforcement/2-Gym/assignment.md",
  "language_code": "da"
}
-->
# Træn Mountain Car

[OpenAI Gym](http://gym.openai.com) er designet på en måde, hvor alle miljøer tilbyder den samme API - dvs. de samme metoder `reset`, `step` og `render`, samt de samme abstraktioner af **aktionsrum** og **observationsrum**. Derfor bør det være muligt at tilpasse de samme forstærkningslæringsalgoritmer til forskellige miljøer med minimale kodeændringer.

## Et Mountain Car-miljø

[Mountain Car-miljøet](https://gym.openai.com/envs/MountainCar-v0/) indeholder en bil, der sidder fast i en dal:

Målet er at komme ud af dalen og fange flaget ved at udføre en af følgende handlinger ved hvert trin:

| Værdi | Betydning |
|---|---|
| 0 | Accelerer til venstre |
| 1 | Ingen acceleration |
| 2 | Accelerer til højre |

Hovedtricket i dette problem er dog, at bilens motor ikke er stærk nok til at bestige bjerget i ét forsøg. Derfor er den eneste måde at lykkes på at køre frem og tilbage for at opbygge momentum.

Observationsrummet består kun af to værdier:

| Num | Observation  | Min | Max |
|-----|--------------|-----|-----|
|  0  | Bilens position | -1.2| 0.6 |
|  1  | Bilens hastighed | -0.07 | 0.07 |

Belønningssystemet for Mountain Car er ret tricky:

 * En belønning på 0 gives, hvis agenten når flaget (position = 0.5) på toppen af bjerget.
 * En belønning på -1 gives, hvis agentens position er mindre end 0.5.

Episoden afsluttes, hvis bilens position er mere end 0.5, eller hvis episodens længde overstiger 200.
## Instruktioner

Tilpas vores forstærkningslæringsalgoritme til at løse Mountain Car-problemet. Start med den eksisterende [notebook.ipynb](../../../../8-Reinforcement/2-Gym/notebook.ipynb)-kode, erstat med det nye miljø, ændr funktionerne til diskretisering af tilstande, og prøv at få den eksisterende algoritme til at træne med minimale kodeændringer. Optimer resultatet ved at justere hyperparametre.

> **Bemærk**: Justering af hyperparametre vil sandsynligvis være nødvendig for at få algoritmen til at konvergere. 
## Vurderingskriterier

| Kriterier | Fremragende | Tilstrækkelig | Kræver forbedring |
| --------- | ----------- | ------------- | ----------------- |
|          | Q-Learning-algoritmen er succesfuldt tilpasset fra CartPole-eksemplet med minimale kodeændringer og er i stand til at løse problemet med at fange flaget på under 200 trin. | En ny Q-Learning-algoritme er blevet adopteret fra internettet, men er veldokumenteret; eller den eksisterende algoritme er tilpasset, men når ikke de ønskede resultater. | Studenten var ikke i stand til at tilpasse nogen algoritme succesfuldt, men har gjort betydelige fremskridt mod løsningen (implementeret tilstands-diskretisering, Q-Table datastruktur osv.) |

---

**Ansvarsfraskrivelse**:  
Dette dokument er blevet oversat ved hjælp af AI-oversættelsestjenesten [Co-op Translator](https://github.com/Azure/co-op-translator). Selvom vi bestræber os på at sikre nøjagtighed, skal det bemærkes, at automatiserede oversættelser kan indeholde fejl eller unøjagtigheder. Det originale dokument på dets oprindelige sprog bør betragtes som den autoritative kilde. For kritisk information anbefales professionel menneskelig oversættelse. Vi påtager os ikke ansvar for eventuelle misforståelser eller fejltolkninger, der måtte opstå som følge af brugen af denne oversættelse.