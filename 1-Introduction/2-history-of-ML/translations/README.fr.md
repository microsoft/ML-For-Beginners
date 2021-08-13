# Histoire du Machine Learning (apprentissage automatique)

![Résumé de l'histoire du machine learning dans un sketchnote](../../../sketchnotes/ml-history.png)
> Sketchnote de [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Quizz préalable](https://white-water-09ec41f0f.azurestaticapps.net/quiz/3?loc=fr)

Dans cette leçon, nous allons parcourir les principales étapes de l'histoire du machine learning et de l'intelligence artificielle.

L'histoire de l'intelligence artificielle, l'IA, en tant que domaine est étroitement liée à l'histoire du machine learning, car les algorithmes et les avancées informatiques qui sous-tendent le ML alimentent le développement de l'IA. Bien que ces domaines en tant que domaines de recherches distincts ont commencé à se cristalliser dans les années 1950, il est important de rappeler que les [découvertes algorithmiques, statistiques, mathématiques, informatiques et techniques](https://wikipedia.org/wiki/Timeline_of_machine_learning) ont précédé et chevauchait cette époque. En fait, le monde réfléchit à ces questions depuis [des centaines d'années](https://fr.wikipedia.org/wiki/Histoire_de_l%27intelligence_artificielle) : cet article traite des fondements intellectuels historiques de l'idée d'une « machine qui pense ».

## Découvertes notables

- 1763, 1812 [théorème de Bayes](https://wikipedia.org/wiki/Bayes%27_theorem) et ses prédécesseurs. Ce théorème et ses applications sous-tendent l'inférence, décrivant la probabilité qu'un événement se produise sur la base de connaissances antérieures.
- 1805 [Théorie des moindres carrés](https://wikipedia.org/wiki/Least_squares) par le mathématicien français Adrien-Marie Legendre. Cette théorie, que vous découvrirez dans notre unité Régression, aide à l'ajustement des données.
- 1913 [Chaînes de Markov](https://wikipedia.org/wiki/Markov_chain) du nom du mathématicien russe Andrey Markov sont utilisées pour décrire une séquence d'événements possibles basée sur un état antérieur.
- 1957 [Perceptron](https://wikipedia.org/wiki/Perceptron) est un type de classificateur linéaire inventé par le psychologue américain Frank Rosenblatt qui sous-tend les progrès de l'apprentissage en profondeur.
- 1967 [Nearest Neighbor](https://wikipedia.org/wiki/Nearest_neighbor) est un algorithme conçu à l'origine pour cartographier les itinéraires. Dans un contexte ML, il est utilisé pour détecter des modèles.
- 1970 [Backpropagation](https://wikipedia.org/wiki/Backpropagation) est utilisé pour former des [réseaux de neurones feedforward (propagation avant)](https://fr.wikipedia.org/wiki/R%C3%A9seau_de_neurones_%C3%A0_propagation_avant).
- 1982 [Réseaux de neurones récurrents](https://wikipedia.org/wiki/Recurrent_neural_network) sont des réseaux de neurones artificiels dérivés de réseaux de neurones à réaction qui créent des graphes temporels.

✅ Faites une petite recherche. Quelles autres dates sont marquantes dans l'histoire du ML et de l'IA ?

## 1950 : Des machines qui pensent

Alan Turing, une personne vraiment remarquable qui a été élue [par le public en 2019](https://wikipedia.org/wiki/Icons:_The_Greatest_Person_of_the_20th_Century) comme le plus grand scientifique du 20e siècle, est reconnu pour avoir aidé à jeter les bases du concept d'une "machine qui peut penser". Il a lutté avec ses opposants et son propre besoin de preuves empiriques de sa théorie en créant le [Test de Turing] (https://www.bbc.com/news/technology-18475646), que vous explorerez dans nos leçons de NLP (TALN en français).

## 1956 : Projet de recherche d'été à Dartmouth

« Le projet de recherche d'été de Dartmouth sur l'intelligence artificielle a été un événement fondateur pour l'intelligence artificielle en tant que domaine », et c'est ici que le terme « intelligence artificielle » a été inventé ([source](https://250.dartmouth.edu/highlights/artificial-intelligence-ai-coined-dartmouth)).

> Chaque aspect de l'apprentissage ou toute autre caractéristique de l'intelligence peut en principe être décrit si précisément qu'une machine peut être conçue pour les simuler.

Le chercheur en tête, le professeur de mathématiques John McCarthy, espérait « procéder sur la base de la conjecture selon laquelle chaque aspect de l'apprentissage ou toute autre caractéristique de l'intelligence peut en principe être décrit avec une telle précision qu'une machine peut être conçue pour les simuler ». Les participants comprenaient une autre sommité dans le domaine, Marvin Minsky.

L'atelier est crédité d'avoir initié et encouragé plusieurs discussions, notamment « l'essor des méthodes symboliques, des systèmes spécialisés sur des domaines limités (premiers systèmes experts) et des systèmes déductifs par rapport aux systèmes inductifs ». ([source](https://fr.wikipedia.org/wiki/Conf%C3%A9rence_de_Dartmouth)).

## 1956 - 1974 : "Les années d'or"

Des années 50 au milieu des années 70, l'optimisme était au rendez-vous en espérant que l'IA puisse résoudre de nombreux problèmes. En 1967, Marvin Minsky a déclaré avec assurance que « Dans une génération... le problème de la création d'"intelligence artificielle" sera substantiellement résolu. » (Minsky, Marvin (1967), Computation: Finite and Infinite Machines, Englewood Cliffs, N.J.: Prentice-Hall)

La recherche sur le Natural Language Processing (traitement du langage naturel en français) a prospéré, la recherche a été affinée et rendue plus puissante, et le concept de « micro-mondes » a été créé, où des tâches simples ont été effectuées en utilisant des instructions en langue naturelle.

La recherche a été bien financée par les agences gouvernementales, des progrès ont été réalisés dans le calcul et les algorithmes, et des prototypes de machines intelligentes ont été construits. Certaines de ces machines incluent :

* [Shakey le robot](https://fr.wikipedia.org/wiki/Shakey_le_robot), qui pouvait manœuvrer et décider comment effectuer des tâches « intelligemment ».

    ![Shakey, un robot intelligent](../images/shakey.jpg)
    > Shaky en 1972

* Eliza, une des premières « chatbot », pouvait converser avec les gens et agir comme une « thérapeute » primitive. Vous en apprendrez plus sur Eliza dans les leçons de NLP.

    ![Eliza, un bot](../images/eliza.png)
    > Une version d'Eliza, un chatbot

* Le « monde des blocs » était un exemple de micro-monde où les blocs pouvaient être empilés et triés, et où des expériences d'apprentissages sur des machines, dans le but qu'elles prennent des décisions, pouvaient être testées. Les avancées réalisées avec des bibliothèques telles que [SHRDLU](https://fr.wikipedia.org/wiki/SHRDLU) ont contribué à faire avancer le natural language processing.

    [![Monde de blocs avec SHRDLU](https://img.youtube.com/vi/QAJz4YKUwqw/0.jpg)](https://www.youtube.com/watch?v=QAJz4YKUwqw "Monde de blocs avec SHRDLU" )
    
    > 🎥 Cliquez sur l'image ci-dessus pour une vidéo : Blocks world with SHRDLU

## 1974 - 1980 : « l'hiver de l'IA »

Au milieu des années 1970, il était devenu évident que la complexité de la fabrication de « machines intelligentes » avait été sous-estimée et que sa promesse, compte tenu de la puissance de calcul disponible, avait été exagérée. Les financements se sont taris et la confiance dans le domaine s'est ralentie. Parmi les problèmes qui ont eu un impact sur la confiance, citons :

- **Restrictions**. La puissance de calcul était trop limitée.
- **Explosion combinatoire**. Le nombre de paramètres à former augmentait de façon exponentielle à mesure que l'on en demandait davantage aux ordinateurs, sans évolution parallèle de la puissance et de la capacité de calcul.
- **Pénurie de données**. Il y avait un manque de données qui a entravé le processus de test, de développement et de raffinement des algorithmes.
- **Posions-nous les bonnes questions ?**. Les questions mêmes, qui étaient posées, ont commencé à être remises en question. Les chercheurs ont commencé à émettre des critiques sur leurs approches :
  - Les tests de Turing ont été remis en question au moyen, entre autres, de la « théorie de la chambre chinoise » qui postulait que « la programmation d'un ordinateur numérique peut faire croire qu'il comprend le langage mais ne peut pas produire une compréhension réelle ». ([source](https://plato.stanford.edu/entries/chinese-room/))
  - L'éthique de l'introduction d'intelligences artificielles telles que la "thérapeute" ELIZA dans la société a été remise en cause.

Dans le même temps, diverses écoles de pensée sur l'IA ont commencé à se former. Une dichotomie a été établie entre les pratiques IA ["scruffy" et "neat"](https://wikipedia.org/wiki/Neats_and_scruffies). Les laboratoires _Scruffy_ peaufinaient leurs programmes pendant des heures jusqu'à ce qu'ils obtiennent les résultats souhaités. Les laboratoires _Neat_ "se concentraient sur la logique et la résolution formelle de problèmes". ELIZA et SHRDLU étaient des systèmes _scruffy_ bien connus. Dans les années 1980, alors qu'émergeait la demande de rendre les systèmes ML reproductibles, l'approche _neat_ a progressivement pris le devant de la scène car ses résultats sont plus explicables.

## 1980 : Systèmes experts

Au fur et à mesure que le domaine s'est développé, ses avantages pour les entreprises sont devenus plus clairs, particulièrement via les « systèmes experts » dans les années 1980. "Les systèmes experts ont été parmi les premières formes vraiment réussies de logiciels d'intelligence artificielle (IA)." ([source](https://fr.wikipedia.org/wiki/Syst%C3%A8me_expert)).

Ce type de système est en fait _hybride_, composé en partie d'un moteur de règles définissant les exigences métier et d'un moteur d'inférence qui exploite le système de règles pour déduire de nouveaux faits.

Cette époque a également vu une attention croissante accordée aux réseaux de neurones.

## 1987 - 1993 : IA « Chill »

La prolifération du matériel spécialisé des systèmes experts a eu pour effet malheureux de devenir trop spécialisée. L'essor des ordinateurs personnels a également concurrencé ces grands systèmes spécialisés et centralisés. La démocratisation de l'informatique a commencé et a finalement ouvert la voie à l'explosion des mégadonnées.

## 1993 - 2011

Cette époque a vu naître une nouvelle ère pour le ML et l'IA afin de résoudre certains des problèmes qui n'avaient pu l'être plus tôt par le manque de données et de puissance de calcul. La quantité de données a commencé à augmenter rapidement et à devenir plus largement disponibles, pour le meilleur et pour le pire, en particulier avec l'avènement du smartphone vers 2007. La puissance de calcul a augmenté de façon exponentielle et les algorithmes ont évolué parallèlement. Le domaine a commencé à gagner en maturité alors que l'ingéniosité a commencé à se cristalliser en une véritable discipline.

## À présent

Aujourd'hui, le machine learning et l'IA touchent presque tous les aspects de notre vie. Cette ère nécessite une compréhension approfondie des risques et des effets potentiels de ces algorithmes sur les vies humaines. Comme l'a déclaré Brad Smith de Microsoft, « les technologies de l'information soulèvent des problèmes qui vont au cœur des protections fondamentales des droits de l'homme comme la vie privée et la liberté d'expression. Ces problèmes accroissent la responsabilité des entreprises technologiques qui créent ces produits. À notre avis, ils appellent également à une réglementation gouvernementale réfléchie et au développement de normes autour des utilisations acceptables" ([source](https://www.technologyreview.com/2019/12/18/102365/the-future-of-ais-impact-on-society/)).

Reste à savoir ce que l'avenir nous réserve, mais il est important de comprendre ces systèmes informatiques ainsi que les logiciels et algorithmes qu'ils exécutent. Nous espérons que ce programme vous aidera à mieux les comprendre afin que vous puissiez décider par vous-même.

[![L'histoire du Deep Learning](https://img.youtube.com/vi/mTtDfKgLm54/0.jpg)](https://www.youtube.com/watch?v=mTtDfKgLm54 "L'histoire du Deep Learning")
> 🎥 Cliquez sur l'image ci-dessus pour une vidéo : Yann LeCun discute de l'histoire du deep learning dans cette conférence

---
## 🚀Challenge

Plongez dans l'un de ces moments historiques et apprenez-en plus sur les personnes derrière ceux-ci. Il y a des personnalités fascinantes, et aucune découverte scientifique n'a jamais été créée avec un vide culturel. Que découvrez-vous ?

## [Quiz de validation des connaissances](https://white-water-09ec41f0f.azurestaticapps.net/quiz/4?loc=fr)

## Révision et auto-apprentissage

Voici quelques articles à regarder et à écouter :

[Ce podcast où Amy Boyd discute de l'évolution de l'IA](http://runasradio.com/Shows/Show/739)

[![L'histoire de l'IA par Amy Boyd](https://img.youtube.com/vi/EJt3_bFYKss/0.jpg)](https://www.youtube.com/watch?v=EJt3_bFYKss "L'histoire de l'IA par Amy Boyd")

## Devoir

[Créer une frise chronologique](assignment.fr.md)
