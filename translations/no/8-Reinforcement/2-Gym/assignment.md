<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "1f2b7441745eb52e25745423b247016b",
  "translation_date": "2025-09-05T22:11:06+00:00",
  "source_file": "8-Reinforcement/2-Gym/assignment.md",
  "language_code": "no"
}
-->
# Tren Mountain Car

[OpenAI Gym](http://gym.openai.com) er designet slik at alle miljøer tilbyr samme API - altså de samme metodene `reset`, `step` og `render`, og de samme abstraksjonene for **handlingsrom** og **observasjonsrom**. Derfor bør det være mulig å tilpasse de samme forsterkningslæringsalgoritmene til forskjellige miljøer med minimale kodeendringer.

## Et Mountain Car-miljø

[Mountain Car-miljøet](https://gym.openai.com/envs/MountainCar-v0/) inneholder en bil som sitter fast i en dal:

Målet er å komme seg ut av dalen og fange flagget, ved å utføre en av følgende handlinger på hvert steg:

| Verdi | Betydning |
|---|---|
| 0 | Akselerer til venstre |
| 1 | Ikke akselerer |
| 2 | Akselerer til høyre |

Hovedutfordringen i dette problemet er imidlertid at bilens motor ikke er sterk nok til å klatre opp fjellet i én enkelt passering. Derfor er den eneste måten å lykkes på å kjøre frem og tilbake for å bygge opp nok moment.

Observasjonsrommet består kun av to verdier:

| Nr | Observasjon  | Min | Maks |
|-----|--------------|-----|-----|
|  0  | Bilens posisjon | -1.2| 0.6 |
|  1  | Bilens hastighet | -0.07 | 0.07 |

Belønningssystemet for Mountain Car er ganske utfordrende:

 * En belønning på 0 gis hvis agenten når flagget (posisjon = 0.5) på toppen av fjellet.
 * En belønning på -1 gis hvis agentens posisjon er mindre enn 0.5.

Episoden avsluttes hvis bilens posisjon er mer enn 0.5, eller hvis episodens lengde overstiger 200.
## Instruksjoner

Tilpass vår forsterkningslæringsalgoritme for å løse Mountain Car-problemet. Start med eksisterende [notebook.ipynb](../../../../8-Reinforcement/2-Gym/notebook.ipynb)-kode, bytt ut miljøet, endre funksjonene for diskretisering av tilstander, og prøv å få den eksisterende algoritmen til å trene med minimale kodeendringer. Optimaliser resultatet ved å justere hyperparametere.

> **Merk**: Justering av hyperparametere vil sannsynligvis være nødvendig for at algoritmen skal konvergere. 
## Vurderingskriterier

| Kriterier | Eksemplarisk | Tilfredsstillende | Trenger forbedring |
| -------- | --------- | -------- | ----------------- |
|          | Q-Learning-algoritmen er vellykket tilpasset fra CartPole-eksempelet, med minimale kodeendringer, og er i stand til å løse problemet med å fange flagget på under 200 steg. | En ny Q-Learning-algoritme er hentet fra Internett, men er godt dokumentert; eller eksisterende algoritme er tilpasset, men oppnår ikke ønskede resultater | Studenten klarte ikke å tilpasse noen algoritme vellykket, men har gjort betydelige fremskritt mot løsningen (implementert tilstands-diskretisering, Q-Tabell datastruktur, osv.) |

---

**Ansvarsfraskrivelse**:  
Dette dokumentet er oversatt ved hjelp av AI-oversettelsestjenesten [Co-op Translator](https://github.com/Azure/co-op-translator). Selv om vi streber etter nøyaktighet, vær oppmerksom på at automatiserte oversettelser kan inneholde feil eller unøyaktigheter. Det originale dokumentet på sitt opprinnelige språk bør anses som den autoritative kilden. For kritisk informasjon anbefales profesjonell menneskelig oversettelse. Vi er ikke ansvarlige for misforståelser eller feiltolkninger som oppstår ved bruk av denne oversettelsen.