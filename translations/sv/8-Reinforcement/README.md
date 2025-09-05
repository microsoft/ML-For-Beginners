<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "20ca019012b1725de956681d036d8b18",
  "translation_date": "2025-09-05T22:01:26+00:00",
  "source_file": "8-Reinforcement/README.md",
  "language_code": "sv"
}
-->
# Introduktion till förstärkningsinlärning

Förstärkningsinlärning, RL, ses som ett av de grundläggande paradigmen inom maskininlärning, vid sidan av övervakad inlärning och oövervakad inlärning. RL handlar om beslut: att fatta rätt beslut eller åtminstone lära sig av dem.

Föreställ dig att du har en simulerad miljö, som aktiemarknaden. Vad händer om du inför en viss reglering? Har det en positiv eller negativ effekt? Om något negativt inträffar måste du ta detta _negativa förstärkning_, lära dig av det och ändra kurs. Om det är ett positivt resultat måste du bygga vidare på den _positiva förstärkningen_.

![peter och vargen](../../../8-Reinforcement/images/peter.png)

> Peter och hans vänner måste fly från den hungriga vargen! Bild av [Jen Looper](https://twitter.com/jenlooper)

## Regionalt ämne: Peter och Vargen (Ryssland)

[Peter och Vargen](https://en.wikipedia.org/wiki/Peter_and_the_Wolf) är en musikalisk saga skriven av den ryske kompositören [Sergej Prokofjev](https://en.wikipedia.org/wiki/Sergei_Prokofiev). Det är en berättelse om den unge pionjären Peter, som modigt lämnar sitt hus och går ut på skogsgläntan för att jaga vargen. I detta avsnitt kommer vi att träna maskininlärningsalgoritmer som hjälper Peter att:

- **Utforska** området runt omkring och bygga en optimal navigeringskarta.
- **Lära sig** att använda en skateboard och balansera på den för att kunna röra sig snabbare.

[![Peter och Vargen](https://img.youtube.com/vi/Fmi5zHg4QSM/0.jpg)](https://www.youtube.com/watch?v=Fmi5zHg4QSM)

> 🎥 Klicka på bilden ovan för att lyssna på Peter och Vargen av Prokofjev

## Förstärkningsinlärning

I tidigare avsnitt har du sett två exempel på maskininlärningsproblem:

- **Övervakad**, där vi har dataset som föreslår exempel på lösningar till problemet vi vill lösa. [Klassificering](../4-Classification/README.md) och [regression](../2-Regression/README.md) är uppgifter inom övervakad inlärning.
- **Oövervakad**, där vi inte har märkta träningsdata. Det främsta exemplet på oövervakad inlärning är [klustring](../5-Clustering/README.md).

I detta avsnitt kommer vi att introducera dig till en ny typ av inlärningsproblem som inte kräver märkta träningsdata. Det finns flera typer av sådana problem:

- **[Semiövervakad inlärning](https://wikipedia.org/wiki/Semi-supervised_learning)**, där vi har mycket omärkta data som kan användas för att förträna modellen.
- **[Förstärkningsinlärning](https://wikipedia.org/wiki/Reinforcement_learning)**, där en agent lär sig hur den ska bete sig genom att utföra experiment i en simulerad miljö.

### Exempel - datorspel

Anta att du vill lära en dator att spela ett spel, som schack eller [Super Mario](https://wikipedia.org/wiki/Super_Mario). För att datorn ska kunna spela ett spel behöver vi att den förutspår vilket drag den ska göra i varje spelstadium. Även om detta kan verka som ett klassificeringsproblem är det inte det – eftersom vi inte har ett dataset med tillstånd och motsvarande åtgärder. Även om vi kanske har viss data, som befintliga schackmatcher eller inspelningar av spelare som spelar Super Mario, är det troligt att dessa data inte täcker ett tillräckligt stort antal möjliga tillstånd.

Istället för att leta efter befintlig speldata bygger **Förstärkningsinlärning** (RL) på idén att *låta datorn spela* många gånger och observera resultatet. För att tillämpa förstärkningsinlärning behöver vi två saker:

- **En miljö** och **en simulator** som låter oss spela spelet många gånger. Denna simulator skulle definiera alla spelregler samt möjliga tillstånd och åtgärder.

- **En belöningsfunktion**, som berättar hur bra vi presterade under varje drag eller spel.

Den största skillnaden mellan andra typer av maskininlärning och RL är att vi i RL vanligtvis inte vet om vi vinner eller förlorar förrän vi avslutar spelet. Därför kan vi inte säga om ett visst drag i sig är bra eller inte – vi får bara en belöning i slutet av spelet. Vårt mål är att designa algoritmer som gör det möjligt för oss att träna en modell under osäkra förhållanden. Vi kommer att lära oss om en RL-algoritm som kallas **Q-learning**.

## Lektioner

1. [Introduktion till förstärkningsinlärning och Q-Learning](1-QLearning/README.md)
2. [Använda en gym-simuleringsmiljö](2-Gym/README.md)

## Krediter

"Introduktion till Förstärkningsinlärning" skrevs med ♥️ av [Dmitry Soshnikov](http://soshnikov.com)

---

**Ansvarsfriskrivning**:  
Detta dokument har översatts med hjälp av AI-översättningstjänsten [Co-op Translator](https://github.com/Azure/co-op-translator). Även om vi strävar efter noggrannhet, bör du vara medveten om att automatiska översättningar kan innehålla fel eller inexaktheter. Det ursprungliga dokumentet på dess originalspråk bör betraktas som den auktoritativa källan. För kritisk information rekommenderas professionell mänsklig översättning. Vi ansvarar inte för eventuella missförstånd eller feltolkningar som uppstår vid användning av denna översättning.