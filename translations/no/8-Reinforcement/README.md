<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "20ca019012b1725de956681d036d8b18",
  "translation_date": "2025-09-05T22:01:42+00:00",
  "source_file": "8-Reinforcement/README.md",
  "language_code": "no"
}
-->
# Introduksjon til forsterkende læring

Forsterkende læring, RL, regnes som en av de grunnleggende paradigmer innen maskinlæring, ved siden av veiledet læring og uveiledet læring. RL handler om beslutninger: å ta riktige beslutninger eller i det minste lære av dem.

Tenk deg at du har et simulert miljø, som aksjemarkedet. Hva skjer hvis du innfører en gitt regulering? Har det en positiv eller negativ effekt? Hvis noe negativt skjer, må du ta denne _negative forsterkningen_, lære av den og endre kurs. Hvis det er et positivt utfall, må du bygge videre på den _positive forsterkningen_.

![Peter og ulven](../../../8-Reinforcement/images/peter.png)

> Peter og vennene hans må unnslippe den sultne ulven! Bilde av [Jen Looper](https://twitter.com/jenlooper)

## Regionalt tema: Peter og ulven (Russland)

[Peter og ulven](https://en.wikipedia.org/wiki/Peter_and_the_Wolf) er et musikalsk eventyr skrevet av den russiske komponisten [Sergej Prokofjev](https://en.wikipedia.org/wiki/Sergei_Prokofiev). Det er en historie om den unge pioneren Peter, som modig går ut av huset sitt til skogkanten for å jage ulven. I denne delen skal vi trene maskinlæringsalgoritmer som vil hjelpe Peter:

- **Utforske** området rundt og bygge et optimalt navigasjonskart.
- **Lære** å bruke et skateboard og balansere på det, for å bevege seg raskere rundt.

[![Peter og ulven](https://img.youtube.com/vi/Fmi5zHg4QSM/0.jpg)](https://www.youtube.com/watch?v=Fmi5zHg4QSM)

> 🎥 Klikk på bildet over for å høre Peter og ulven av Prokofjev

## Forsterkende læring

I tidligere deler har du sett to eksempler på maskinlæringsproblemer:

- **Veiledet**, der vi har datasett som foreslår eksempler på løsninger til problemet vi ønsker å løse. [Klassifisering](../4-Classification/README.md) og [regresjon](../2-Regression/README.md) er oppgaver innen veiledet læring.
- **Uveiledet**, der vi ikke har merkede treningsdata. Hovedeksempelet på uveiledet læring er [Clustering](../5-Clustering/README.md).

I denne delen vil vi introdusere deg for en ny type læringsproblem som ikke krever merkede treningsdata. Det finnes flere typer slike problemer:

- **[Semi-veiledet læring](https://wikipedia.org/wiki/Semi-supervised_learning)**, der vi har mye umerkede data som kan brukes til å forhåndstrene modellen.
- **[Forsterkende læring](https://wikipedia.org/wiki/Reinforcement_learning)**, der en agent lærer hvordan den skal oppføre seg ved å utføre eksperimenter i et simulert miljø.

### Eksempel - dataspill

Anta at du vil lære en datamaskin å spille et spill, som sjakk eller [Super Mario](https://wikipedia.org/wiki/Super_Mario). For at datamaskinen skal spille et spill, må den forutsi hvilken handling den skal ta i hver spilltilstand. Selv om dette kan virke som et klassifiseringsproblem, er det ikke det - fordi vi ikke har et datasett med tilstander og tilsvarende handlinger. Selv om vi kanskje har noen data, som eksisterende sjakkpartier eller opptak av spillere som spiller Super Mario, er det sannsynlig at disse dataene ikke dekker et stort nok antall mulige tilstander.

I stedet for å lete etter eksisterende spilldata, er **Forsterkende læring** (RL) basert på ideen om *å få datamaskinen til å spille* mange ganger og observere resultatet. For å bruke forsterkende læring trenger vi to ting:

- **Et miljø** og **en simulator** som lar oss spille et spill mange ganger. Denne simulatoren vil definere alle spillregler samt mulige tilstander og handlinger.

- **En belønningsfunksjon**, som forteller oss hvor godt vi gjorde det under hver handling eller spill.

Den største forskjellen mellom andre typer maskinlæring og RL er at i RL vet vi vanligvis ikke om vi vinner eller taper før vi er ferdige med spillet. Dermed kan vi ikke si om en bestemt handling alene er god eller ikke - vi mottar bare en belønning ved slutten av spillet. Målet vårt er å designe algoritmer som lar oss trene en modell under usikre forhold. Vi skal lære om en RL-algoritme kalt **Q-læring**.

## Leksjoner

1. [Introduksjon til forsterkende læring og Q-læring](1-QLearning/README.md)
2. [Bruke et gymsimuleringsmiljø](2-Gym/README.md)

## Kreditering

"Introduksjon til forsterkende læring" ble skrevet med ♥️ av [Dmitry Soshnikov](http://soshnikov.com)

---

**Ansvarsfraskrivelse**:  
Dette dokumentet er oversatt ved hjelp av AI-oversettelsestjenesten [Co-op Translator](https://github.com/Azure/co-op-translator). Selv om vi streber etter nøyaktighet, vær oppmerksom på at automatiserte oversettelser kan inneholde feil eller unøyaktigheter. Det originale dokumentet på sitt opprinnelige språk bør anses som den autoritative kilden. For kritisk informasjon anbefales profesjonell menneskelig oversettelse. Vi er ikke ansvarlige for misforståelser eller feiltolkninger som oppstår ved bruk av denne oversettelsen.