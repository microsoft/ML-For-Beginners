<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "20ca019012b1725de956681d036d8b18",
  "translation_date": "2025-09-05T01:03:59+00:00",
  "source_file": "8-Reinforcement/README.md",
  "language_code": "da"
}
-->
# Introduktion til forstærkningslæring

Forstærkningslæring, RL, betragtes som en af de grundlæggende paradigmer inden for maskinlæring, ved siden af superviseret læring og usuperviseret læring. RL handler om beslutninger: at træffe de rigtige beslutninger eller i det mindste lære af dem.

Forestil dig, at du har et simuleret miljø, som f.eks. aktiemarkedet. Hvad sker der, hvis du indfører en given regulering? Har det en positiv eller negativ effekt? Hvis noget negativt sker, skal du tage denne _negative forstærkning_, lære af den og ændre kurs. Hvis det er et positivt resultat, skal du bygge videre på den _positive forstærkning_.

![Peter og ulven](../../../8-Reinforcement/images/peter.png)

> Peter og hans venner skal undslippe den sultne ulv! Billede af [Jen Looper](https://twitter.com/jenlooper)

## Regionalt emne: Peter og ulven (Rusland)

[Peter og ulven](https://en.wikipedia.org/wiki/Peter_and_the_Wolf) er et musikalsk eventyr skrevet af den russiske komponist [Sergei Prokofiev](https://en.wikipedia.org/wiki/Sergei_Prokofiev). Det er en historie om den unge pioner Peter, der modigt går ud af sit hus til lysningen i skoven for at jage ulven. I denne sektion vil vi træne maskinlæringsalgoritmer, der kan hjælpe Peter:

- **Udforske** det omkringliggende område og opbygge et optimalt navigationskort.
- **Lære** at bruge et skateboard og balancere på det for at bevæge sig hurtigere rundt.

[![Peter og ulven](https://img.youtube.com/vi/Fmi5zHg4QSM/0.jpg)](https://www.youtube.com/watch?v=Fmi5zHg4QSM)

> 🎥 Klik på billedet ovenfor for at lytte til Peter og ulven af Prokofiev

## Forstærkningslæring

I tidligere sektioner har du set to eksempler på maskinlæringsproblemer:

- **Superviseret**, hvor vi har datasæt, der foreslår eksempler på løsninger til det problem, vi ønsker at løse. [Klassifikation](../4-Classification/README.md) og [regression](../2-Regression/README.md) er superviserede læringsopgaver.
- **Usuperviseret**, hvor vi ikke har mærkede træningsdata. Det primære eksempel på usuperviseret læring er [Clustering](../5-Clustering/README.md).

I denne sektion vil vi introducere dig til en ny type læringsproblem, der ikke kræver mærkede træningsdata. Der er flere typer af sådanne problemer:

- **[Semi-superviseret læring](https://wikipedia.org/wiki/Semi-supervised_learning)**, hvor vi har en masse umærkede data, der kan bruges til at fortræne modellen.
- **[Forstærkningslæring](https://wikipedia.org/wiki/Reinforcement_learning)**, hvor en agent lærer at opføre sig ved at udføre eksperimenter i et simuleret miljø.

### Eksempel - computerspil

Forestil dig, at du vil lære en computer at spille et spil, som f.eks. skak eller [Super Mario](https://wikipedia.org/wiki/Super_Mario). For at computeren kan spille et spil, skal den kunne forudsige, hvilket træk den skal foretage i hver af spillets tilstande. Selvom dette kan virke som et klassifikationsproblem, er det det ikke - fordi vi ikke har et datasæt med tilstande og tilsvarende handlinger. Selvom vi måske har nogle data som eksisterende skakpartier eller optagelser af spillere, der spiller Super Mario, er det sandsynligt, at disse data ikke tilstrækkeligt dækker et stort nok antal mulige tilstande.

I stedet for at lede efter eksisterende spildata er **Forstærkningslæring** (RL) baseret på ideen om *at lade computeren spille* mange gange og observere resultatet. For at anvende Forstærkningslæring har vi brug for to ting:

- **Et miljø** og **en simulator**, der giver os mulighed for at spille et spil mange gange. Denne simulator skal definere alle spillets regler samt mulige tilstande og handlinger.

- **En belønningsfunktion**, der fortæller os, hvor godt vi klarede os under hvert træk eller spil.

Den største forskel mellem andre typer maskinlæring og RL er, at vi i RL typisk ikke ved, om vi vinder eller taber, før vi har afsluttet spillet. Derfor kan vi ikke sige, om et bestemt træk alene er godt eller ej - vi modtager kun en belønning ved slutningen af spillet. Vores mål er at designe algoritmer, der gør det muligt for os at træne en model under usikre forhold. Vi vil lære om en RL-algoritme kaldet **Q-learning**.

## Lektioner

1. [Introduktion til forstærkningslæring og Q-Learning](1-QLearning/README.md)
2. [Brug af et gym-simuleringsmiljø](2-Gym/README.md)

## Credits

"Introduktion til Forstærkningslæring" er skrevet med ♥️ af [Dmitry Soshnikov](http://soshnikov.com)

---

**Ansvarsfraskrivelse**:  
Dette dokument er blevet oversat ved hjælp af AI-oversættelsestjenesten [Co-op Translator](https://github.com/Azure/co-op-translator). Selvom vi bestræber os på nøjagtighed, skal du være opmærksom på, at automatiserede oversættelser kan indeholde fejl eller unøjagtigheder. Det originale dokument på dets oprindelige sprog bør betragtes som den autoritative kilde. For kritisk information anbefales professionel menneskelig oversættelse. Vi er ikke ansvarlige for eventuelle misforståelser eller fejltolkninger, der opstår som følge af brugen af denne oversættelse.