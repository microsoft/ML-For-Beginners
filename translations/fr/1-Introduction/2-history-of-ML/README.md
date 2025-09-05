<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "6a05fec147e734c3e6bfa54505648e2b",
  "translation_date": "2025-09-04T23:01:16+00:00",
  "source_file": "1-Introduction/2-history-of-ML/README.md",
  "language_code": "fr"
}
-->
# Histoire de l'apprentissage automatique

![Résumé de l'histoire de l'apprentissage automatique sous forme de sketchnote](../../../../sketchnotes/ml-history.png)
> Sketchnote par [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Quiz avant le cours](https://ff-quizzes.netlify.app/en/ml/)

---

[![ML pour débutants - Histoire de l'apprentissage automatique](https://img.youtube.com/vi/N6wxM4wZ7V0/0.jpg)](https://youtu.be/N6wxM4wZ7V0 "ML pour débutants - Histoire de l'apprentissage automatique")

> 🎥 Cliquez sur l'image ci-dessus pour une courte vidéo sur cette leçon.

Dans cette leçon, nous allons parcourir les étapes majeures de l'histoire de l'apprentissage automatique et de l'intelligence artificielle.

L'histoire de l'intelligence artificielle (IA) en tant que domaine est étroitement liée à celle de l'apprentissage automatique, car les algorithmes et les avancées computationnelles qui sous-tendent l'apprentissage automatique ont alimenté le développement de l'IA. Il est utile de se rappeler que, bien que ces domaines en tant qu'aires d'étude distinctes aient commencé à se cristalliser dans les années 1950, des découvertes importantes [algorithmiques, statistiques, mathématiques, computationnelles et techniques](https://wikipedia.org/wiki/Timeline_of_machine_learning) ont précédé et chevauché cette époque. En fait, les gens réfléchissent à ces questions depuis [des centaines d'années](https://wikipedia.org/wiki/History_of_artificial_intelligence) : cet article discute des fondements intellectuels historiques de l'idée d'une "machine pensante".

---
## Découvertes notables

- 1763, 1812 [Théorème de Bayes](https://wikipedia.org/wiki/Bayes%27_theorem) et ses prédécesseurs. Ce théorème et ses applications sous-tendent l'inférence, décrivant la probabilité qu'un événement se produise en fonction de connaissances préalables.
- 1805 [Théorie des moindres carrés](https://wikipedia.org/wiki/Least_squares) par le mathématicien français Adrien-Marie Legendre. Cette théorie, que vous apprendrez dans notre unité sur la régression, aide à ajuster les données.
- 1913 [Chaînes de Markov](https://wikipedia.org/wiki/Markov_chain), nommées d'après le mathématicien russe Andrey Markov, sont utilisées pour décrire une séquence d'événements possibles basée sur un état précédent.
- 1957 [Perceptron](https://wikipedia.org/wiki/Perceptron), un type de classificateur linéaire inventé par le psychologue américain Frank Rosenblatt, qui sous-tend les avancées en apprentissage profond.

---

- 1967 [Plus proche voisin](https://wikipedia.org/wiki/Nearest_neighbor), un algorithme initialement conçu pour cartographier des itinéraires. Dans un contexte d'apprentissage automatique, il est utilisé pour détecter des motifs.
- 1970 [Rétropropagation](https://wikipedia.org/wiki/Backpropagation), utilisée pour entraîner les [réseaux neuronaux feedforward](https://wikipedia.org/wiki/Feedforward_neural_network).
- 1982 [Réseaux neuronaux récurrents](https://wikipedia.org/wiki/Recurrent_neural_network), des réseaux neuronaux artificiels dérivés des réseaux neuronaux feedforward qui créent des graphes temporels.

✅ Faites quelques recherches. Quelles autres dates se démarquent comme étant cruciales dans l'histoire de l'apprentissage automatique et de l'IA ?

---
## 1950 : Des machines qui pensent

Alan Turing, une personne véritablement remarquable qui a été élu [par le public en 2019](https://wikipedia.org/wiki/Icons:_The_Greatest_Person_of_the_20th_Century) comme le plus grand scientifique du 20e siècle, est crédité d'avoir contribué à poser les bases du concept d'une "machine capable de penser". Il a affronté les sceptiques et son propre besoin de preuves empiriques de ce concept en partie en créant le [Test de Turing](https://www.bbc.com/news/technology-18475646), que vous explorerez dans nos leçons sur le traitement du langage naturel.

---
## 1956 : Projet de recherche d'été à Dartmouth

"Le projet de recherche d'été à Dartmouth sur l'intelligence artificielle a été un événement fondateur pour l'intelligence artificielle en tant que domaine", et c'est ici que le terme "intelligence artificielle" a été inventé ([source](https://250.dartmouth.edu/highlights/artificial-intelligence-ai-coined-dartmouth)).

> Chaque aspect de l'apprentissage ou toute autre caractéristique de l'intelligence peut en principe être décrit si précisément qu'une machine peut être conçue pour le simuler.

---

Le chercheur principal, le professeur de mathématiques John McCarthy, espérait "progresser sur la base de l'hypothèse que chaque aspect de l'apprentissage ou toute autre caractéristique de l'intelligence peut en principe être décrit si précisément qu'une machine peut être conçue pour le simuler." Les participants comprenaient un autre grand nom du domaine, Marvin Minsky.

L'atelier est crédité d'avoir initié et encouragé plusieurs discussions, notamment "l'essor des méthodes symboliques, des systèmes axés sur des domaines limités (premiers systèmes experts) et des systèmes déductifs par rapport aux systèmes inductifs." ([source](https://wikipedia.org/wiki/Dartmouth_workshop)).

---
## 1956 - 1974 : "Les années dorées"

Des années 1950 au milieu des années 1970, l'optimisme était élevé quant à la capacité de l'IA à résoudre de nombreux problèmes. En 1967, Marvin Minsky déclarait avec confiance : "Dans une génération... le problème de la création de 'l'intelligence artificielle' sera substantiellement résolu." (Minsky, Marvin (1967), Computation: Finite and Infinite Machines, Englewood Cliffs, N.J.: Prentice-Hall)

La recherche en traitement du langage naturel a prospéré, la recherche a été affinée et rendue plus puissante, et le concept de "micro-mondes" a été créé, où des tâches simples étaient accomplies en utilisant des instructions en langage clair.

---

La recherche était bien financée par des agences gouvernementales, des avancées ont été réalisées en computation et en algorithmes, et des prototypes de machines intelligentes ont été construits. Certaines de ces machines incluent :

* [Shakey le robot](https://wikipedia.org/wiki/Shakey_the_robot), qui pouvait se déplacer et décider comment effectuer des tâches "intelligemment".

    ![Shakey, un robot intelligent](../../../../1-Introduction/2-history-of-ML/images/shakey.jpg)
    > Shakey en 1972

---

* Eliza, un premier "chatterbot", pouvait converser avec les gens et agir comme un "thérapeute" primitif. Vous en apprendrez davantage sur Eliza dans les leçons sur le traitement du langage naturel.

    ![Eliza, un bot](../../../../1-Introduction/2-history-of-ML/images/eliza.png)
    > Une version d'Eliza, un chatbot

---

* "Blocks world" était un exemple de micro-monde où des blocs pouvaient être empilés et triés, et des expériences pour enseigner aux machines à prendre des décisions pouvaient être testées. Les avancées construites avec des bibliothèques telles que [SHRDLU](https://wikipedia.org/wiki/SHRDLU) ont aidé à propulser le traitement du langage.

    [![blocks world avec SHRDLU](https://img.youtube.com/vi/QAJz4YKUwqw/0.jpg)](https://www.youtube.com/watch?v=QAJz4YKUwqw "blocks world avec SHRDLU")

    > 🎥 Cliquez sur l'image ci-dessus pour une vidéo : Blocks world avec SHRDLU

---
## 1974 - 1980 : "Hiver de l'IA"

Au milieu des années 1970, il était devenu évident que la complexité de la création de "machines intelligentes" avait été sous-estimée et que ses promesses, compte tenu de la puissance de calcul disponible, avaient été exagérées. Le financement s'est tari et la confiance dans le domaine a ralenti. Certains problèmes qui ont affecté la confiance incluent :
---
- **Limitations**. La puissance de calcul était trop limitée.
- **Explosion combinatoire**. Le nombre de paramètres nécessaires à l'entraînement augmentait de manière exponentielle à mesure que l'on demandait plus aux ordinateurs, sans une évolution parallèle de la puissance et des capacités de calcul.
- **Manque de données**. Il y avait un manque de données qui entravait le processus de test, de développement et de raffinement des algorithmes.
- **Posons-nous les bonnes questions ?**. Les questions mêmes qui étaient posées ont commencé à être remises en question. Les chercheurs ont commencé à recevoir des critiques sur leurs approches :
  - Les tests de Turing ont été remis en question, entre autres idées, par la théorie de la "chambre chinoise" qui postulait que "programmer un ordinateur numérique peut lui donner l'apparence de comprendre le langage mais ne pourrait pas produire une véritable compréhension." ([source](https://plato.stanford.edu/entries/chinese-room/))
  - L'éthique de l'introduction d'intelligences artificielles telles que le "thérapeute" ELIZA dans la société a été contestée.

---

En même temps, diverses écoles de pensée en IA ont commencé à se former. Une dichotomie s'est établie entre les pratiques ["scruffy" vs. "neat AI"](https://wikipedia.org/wiki/Neats_and_scruffies). Les laboratoires _scruffy_ ajustaient les programmes pendant des heures jusqu'à obtenir les résultats souhaités. Les laboratoires _neat_ "se concentraient sur la logique et la résolution formelle de problèmes". ELIZA et SHRDLU étaient des systèmes _scruffy_ bien connus. Dans les années 1980, à mesure que la demande pour rendre les systèmes d'apprentissage automatique reproductibles augmentait, l'approche _neat_ a progressivement pris le devant de la scène car ses résultats sont plus explicables.

---
## Années 1980 : Systèmes experts

À mesure que le domaine grandissait, ses avantages pour les entreprises devenaient plus clairs, et dans les années 1980, la prolifération des "systèmes experts" s'est également accrue. "Les systèmes experts étaient parmi les premières formes véritablement réussies de logiciels d'intelligence artificielle (IA)." ([source](https://wikipedia.org/wiki/Expert_system)).

Ce type de système est en réalité _hybride_, consistant en partie en un moteur de règles définissant les exigences métier, et un moteur d'inférence qui exploitait le système de règles pour déduire de nouveaux faits.

Cette époque a également vu une attention croissante portée aux réseaux neuronaux.

---
## 1987 - 1993 : "Refroidissement de l'IA"

La prolifération du matériel spécialisé pour les systèmes experts a eu l'effet malheureux de devenir trop spécialisée. L'essor des ordinateurs personnels a également concurrencé ces grands systèmes spécialisés et centralisés. La démocratisation de l'informatique avait commencé, et elle a finalement ouvert la voie à l'explosion moderne des big data.

---
## 1993 - 2011

Cette époque a vu une nouvelle ère pour l'apprentissage automatique et l'IA, permettant de résoudre certains des problèmes causés auparavant par le manque de données et de puissance de calcul. La quantité de données a commencé à augmenter rapidement et à devenir plus largement disponible, pour le meilleur et pour le pire, notamment avec l'avènement du smartphone vers 2007. La puissance de calcul a augmenté de manière exponentielle, et les algorithmes ont évolué en parallèle. Le domaine a commencé à gagner en maturité à mesure que les jours libres du passé se cristallisaient en une véritable discipline.

---
## Aujourd'hui

Aujourd'hui, l'apprentissage automatique et l'IA touchent presque toutes les parties de nos vies. Cette époque appelle à une compréhension attentive des risques et des effets potentiels de ces algorithmes sur les vies humaines. Comme l'a déclaré Brad Smith de Microsoft : "La technologie de l'information soulève des questions qui touchent au cœur des protections fondamentales des droits humains comme la vie privée et la liberté d'expression. Ces questions augmentent la responsabilité des entreprises technologiques qui créent ces produits. À notre avis, elles appellent également à une réglementation gouvernementale réfléchie et au développement de normes autour des utilisations acceptables" ([source](https://www.technologyreview.com/2019/12/18/102365/the-future-of-ais-impact-on-society/)).

---

Il reste à voir ce que l'avenir nous réserve, mais il est important de comprendre ces systèmes informatiques ainsi que les logiciels et algorithmes qu'ils exécutent. Nous espérons que ce programme vous aidera à mieux comprendre afin que vous puissiez décider par vous-même.

[![L'histoire de l'apprentissage profond](https://img.youtube.com/vi/mTtDfKgLm54/0.jpg)](https://www.youtube.com/watch?v=mTtDfKgLm54 "L'histoire de l'apprentissage profond")
> 🎥 Cliquez sur l'image ci-dessus pour une vidéo : Yann LeCun discute de l'histoire de l'apprentissage profond dans cette conférence

---
## 🚀Défi

Plongez dans l'un de ces moments historiques et apprenez-en davantage sur les personnes derrière eux. Il y a des personnages fascinants, et aucune découverte scientifique n'a jamais été créée dans un vide culturel. Que découvrez-vous ?

## [Quiz après le cours](https://ff-quizzes.netlify.app/en/ml/)

---
## Révision et étude personnelle

Voici des éléments à regarder et écouter :

[Ce podcast où Amy Boyd discute de l'évolution de l'IA](http://runasradio.com/Shows/Show/739)

[![L'histoire de l'IA par Amy Boyd](https://img.youtube.com/vi/EJt3_bFYKss/0.jpg)](https://www.youtube.com/watch?v=EJt3_bFYKss "L'histoire de l'IA par Amy Boyd")

---

## Devoir

[Créez une chronologie](assignment.md)

---

**Avertissement** :  
Ce document a été traduit à l'aide du service de traduction automatique [Co-op Translator](https://github.com/Azure/co-op-translator). Bien que nous nous efforcions d'assurer l'exactitude, veuillez noter que les traductions automatisées peuvent contenir des erreurs ou des inexactitudes. Le document original dans sa langue d'origine doit être considéré comme la source faisant autorité. Pour des informations critiques, il est recommandé de recourir à une traduction professionnelle réalisée par un humain. Nous déclinons toute responsabilité en cas de malentendus ou d'interprétations erronées résultant de l'utilisation de cette traduction.