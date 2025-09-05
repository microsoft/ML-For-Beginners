<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "20ca019012b1725de956681d036d8b18",
  "translation_date": "2025-09-05T20:09:22+00:00",
  "source_file": "8-Reinforcement/README.md",
  "language_code": "nl"
}
-->
# Introductie tot reinforcement learning

Reinforcement learning, RL, wordt gezien als een van de fundamentele machine learning paradigma's, naast supervised learning en unsupervised learning. RL draait om beslissingen: het nemen van de juiste beslissingen of er in ieder geval van leren.

Stel je een gesimuleerde omgeving voor, zoals de aandelenmarkt. Wat gebeurt er als je een bepaalde regelgeving oplegt? Heeft het een positief of negatief effect? Als er iets negatiefs gebeurt, moet je deze _negatieve versterking_ gebruiken, ervan leren en van koers veranderen. Als het een positief resultaat is, moet je voortbouwen op die _positieve versterking_.

![peter en de wolf](../../../8-Reinforcement/images/peter.png)

> Peter en zijn vrienden moeten ontsnappen aan de hongerige wolf! Afbeelding door [Jen Looper](https://twitter.com/jenlooper)

## Regionaal thema: Peter en de Wolf (Rusland)

[Peter en de Wolf](https://en.wikipedia.org/wiki/Peter_and_the_Wolf) is een muzikaal sprookje geschreven door de Russische componist [Sergei Prokofiev](https://en.wikipedia.org/wiki/Sergei_Prokofiev). Het is een verhaal over de jonge pionier Peter, die dapper zijn huis verlaat om in de bosweide de wolf te achtervolgen. In deze sectie zullen we machine learning-algoritmes trainen die Peter kunnen helpen:

- **Verkennen** van de omgeving en het bouwen van een optimale navigatiekaart.
- **Leren** hoe hij een skateboard kan gebruiken en erop kan balanceren, zodat hij zich sneller kan verplaatsen.

[![Peter en de Wolf](https://img.youtube.com/vi/Fmi5zHg4QSM/0.jpg)](https://www.youtube.com/watch?v=Fmi5zHg4QSM)

> 🎥 Klik op de afbeelding hierboven om te luisteren naar Peter en de Wolf van Prokofiev.

## Reinforcement learning

In eerdere secties heb je twee voorbeelden van machine learning-problemen gezien:

- **Supervised**, waarbij we datasets hebben die voorbeeldoplossingen suggereren voor het probleem dat we willen oplossen. [Classificatie](../4-Classification/README.md) en [regressie](../2-Regression/README.md) zijn supervised learning taken.
- **Unsupervised**, waarbij we geen gelabelde trainingsdata hebben. Het belangrijkste voorbeeld van unsupervised learning is [Clustering](../5-Clustering/README.md).

In deze sectie introduceren we een nieuw type leerprobleem dat geen gelabelde trainingsdata vereist. Er zijn verschillende soorten van dergelijke problemen:

- **[Semi-supervised learning](https://wikipedia.org/wiki/Semi-supervised_learning)**, waarbij we veel niet-gelabelde data hebben die kan worden gebruikt om het model vooraf te trainen.
- **[Reinforcement learning](https://wikipedia.org/wiki/Reinforcement_learning)**, waarbij een agent leert hoe hij zich moet gedragen door experimenten uit te voeren in een gesimuleerde omgeving.

### Voorbeeld - computerspel

Stel dat je een computer wilt leren een spel te spelen, zoals schaken of [Super Mario](https://wikipedia.org/wiki/Super_Mario). Om de computer een spel te laten spelen, moeten we hem laten voorspellen welke zet hij moet doen in elke spelstatus. Hoewel dit misschien een classificatieprobleem lijkt, is het dat niet - omdat we geen dataset hebben met statussen en bijbehorende acties. Hoewel we mogelijk gegevens hebben zoals bestaande schaakpartijen of opnames van spelers die Super Mario spelen, is het waarschijnlijk dat die gegevens niet voldoende een groot aantal mogelijke statussen dekken.

In plaats van te zoeken naar bestaande spelgegevens, is **Reinforcement Learning** (RL) gebaseerd op het idee van *de computer vaak laten spelen* en het resultaat observeren. Om Reinforcement Learning toe te passen, hebben we twee dingen nodig:

- **Een omgeving** en **een simulator** die ons in staat stellen een spel vaak te spelen. Deze simulator zou alle spelregels evenals mogelijke statussen en acties definiëren.

- **Een beloningsfunctie**, die ons vertelt hoe goed we het hebben gedaan tijdens elke zet of elk spel.

Het belangrijkste verschil tussen andere soorten machine learning en RL is dat we bij RL meestal niet weten of we winnen of verliezen totdat we het spel hebben voltooid. We kunnen dus niet zeggen of een bepaalde zet op zichzelf goed is of niet - we ontvangen pas een beloning aan het einde van het spel. Ons doel is om algoritmes te ontwerpen die ons in staat stellen een model te trainen onder onzekere omstandigheden. We zullen leren over een RL-algoritme genaamd **Q-learning**.

## Lessen

1. [Introductie tot reinforcement learning en Q-Learning](1-QLearning/README.md)
2. [Gebruik van een gym-simulatieomgeving](2-Gym/README.md)

## Credits

"Introductie tot Reinforcement Learning" is geschreven met ♥️ door [Dmitry Soshnikov](http://soshnikov.com)

---

**Disclaimer**:  
Dit document is vertaald met behulp van de AI-vertalingsservice [Co-op Translator](https://github.com/Azure/co-op-translator). Hoewel we ons best doen om nauwkeurigheid te garanderen, dient u zich ervan bewust te zijn dat geautomatiseerde vertalingen fouten of onnauwkeurigheden kunnen bevatten. Het originele document in zijn oorspronkelijke taal moet worden beschouwd als de gezaghebbende bron. Voor kritieke informatie wordt professionele menselijke vertaling aanbevolen. Wij zijn niet aansprakelijk voor misverstanden of verkeerde interpretaties die voortvloeien uit het gebruik van deze vertaling.