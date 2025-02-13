# Postscript : L'apprentissage automatique dans le monde réel

![Résumé de l'apprentissage automatique dans le monde réel dans un sketchnote](../../../../translated_images/ml-realworld.26ee2746716155771f8076598b6145e6533fe4a9e2e465ea745f46648cbf1b84.fr.png)
> Sketchnote par [Tomomi Imura](https://www.twitter.com/girlie_mac)

Dans ce programme, vous avez appris de nombreuses façons de préparer des données pour l'entraînement et de créer des modèles d'apprentissage automatique. Vous avez construit une série de modèles classiques de régression, de clustering, de classification, de traitement du langage naturel et de séries temporelles. Félicitations ! Maintenant, vous vous demandez peut-être à quoi cela sert... quelles sont les applications réelles de ces modèles ?

Bien qu'un grand intérêt de l'industrie ait été suscité par l'IA, qui utilise généralement l'apprentissage profond, il existe encore des applications précieuses pour les modèles d'apprentissage automatique classiques. Vous pourriez même utiliser certaines de ces applications aujourd'hui ! Dans cette leçon, vous explorerez comment huit secteurs différents et domaines d'expertise utilisent ces types de modèles pour rendre leurs applications plus performantes, fiables, intelligentes et précieuses pour les utilisateurs.

## [Quiz avant la leçon](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/49/)

## 💰 Finance

Le secteur financier offre de nombreuses opportunités pour l'apprentissage automatique. De nombreux problèmes dans ce domaine peuvent être modélisés et résolus en utilisant l'apprentissage automatique.

### Détection de fraude par carte de crédit

Nous avons appris à propos du [clustering k-means](../../5-Clustering/2-K-Means/README.md) plus tôt dans le cours, mais comment peut-il être utilisé pour résoudre des problèmes liés à la fraude par carte de crédit ?

Le clustering k-means est utile lors d'une technique de détection de fraude par carte de crédit appelée **détection d'anomalies**. Les anomalies, ou écarts dans les observations d'un ensemble de données, peuvent nous indiquer si une carte de crédit est utilisée normalement ou si quelque chose d'inhabituel se produit. Comme le montre l'article lié ci-dessous, vous pouvez trier les données de carte de crédit en utilisant un algorithme de clustering k-means et attribuer chaque transaction à un cluster en fonction de son caractère d'anomalie. Ensuite, vous pouvez évaluer les clusters les plus risqués pour les transactions frauduleuses par rapport aux transactions légitimes.
[Référence](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.680.1195&rep=rep1&type=pdf)

### Gestion de patrimoine

Dans la gestion de patrimoine, un individu ou une entreprise gère des investissements au nom de ses clients. Leur travail consiste à maintenir et à accroître la richesse à long terme, il est donc essentiel de choisir des investissements qui performe bien.

Une façon d'évaluer la performance d'un investissement particulier est à travers la régression statistique. La [régression linéaire](../../2-Regression/1-Tools/README.md) est un outil précieux pour comprendre comment un fonds performe par rapport à un certain indice de référence. Nous pouvons également déduire si les résultats de la régression sont statistiquement significatifs ou dans quelle mesure ils affecteraient les investissements d'un client. Vous pourriez même approfondir votre analyse en utilisant la régression multiple, où des facteurs de risque supplémentaires peuvent être pris en compte. Pour un exemple de la façon dont cela fonctionnerait pour un fonds spécifique, consultez l'article ci-dessous sur l'évaluation de la performance des fonds à l'aide de la régression.
[Référence](http://www.brightwoodventures.com/evaluating-fund-performance-using-regression/)

## 🎓 Éducation

Le secteur éducatif est également un domaine très intéressant où l'apprentissage automatique peut être appliqué. Il existe des problèmes intéressants à résoudre, comme la détection de tricheries lors des tests ou des essais, ou la gestion des biais, qu'ils soient intentionnels ou non, dans le processus de correction.

### Prédiction du comportement des étudiants

[Coursera](https://coursera.com), un fournisseur de cours en ligne, a un excellent blog technique où ils discutent de nombreuses décisions d'ingénierie. Dans cette étude de cas, ils ont tracé une ligne de régression pour essayer d'explorer toute corrélation entre un faible score NPS (Net Promoter Score) et la rétention ou l'abandon des cours.
[Référence](https://medium.com/coursera-engineering/controlled-regression-quantifying-the-impact-of-course-quality-on-learner-retention-31f956bd592a)

### Atténuation des biais

[Grammarly](https://grammarly.com), un assistant d'écriture qui vérifie les erreurs d'orthographe et de grammaire, utilise des systèmes sophistiqués de [traitement du langage naturel](../../6-NLP/README.md) dans ses produits. Ils ont publié une étude de cas intéressante dans leur blog technique sur la manière dont ils ont traité le biais de genre dans l'apprentissage automatique, dont vous avez entendu parler dans notre [leçon d'introduction à l'équité](../../1-Introduction/3-fairness/README.md).
[Référence](https://www.grammarly.com/blog/engineering/mitigating-gender-bias-in-autocorrect/)

## 👜 Vente au détail

Le secteur de la vente au détail peut certainement bénéficier de l'utilisation de l'apprentissage automatique, que ce soit pour créer une meilleure expérience client ou pour gérer les stocks de manière optimale.

### Personnalisation du parcours client

Chez Wayfair, une entreprise qui vend des articles pour la maison comme des meubles, aider les clients à trouver les bons produits en fonction de leurs goûts et de leurs besoins est primordial. Dans cet article, des ingénieurs de l'entreprise décrivent comment ils utilisent l'apprentissage automatique et le traitement du langage naturel pour "afficher les bons résultats pour les clients". Notamment, leur moteur d'intention de requête a été conçu pour utiliser l'extraction d'entités, l'entraînement de classificateurs, l'extraction d'actifs et d'opinions, ainsi que le marquage de sentiment sur les avis des clients. C'est un cas classique de la façon dont le traitement du langage naturel fonctionne dans le commerce en ligne.
[Référence](https://www.aboutwayfair.com/tech-innovation/how-we-use-machine-learning-and-natural-language-processing-to-empower-search)

### Gestion des stocks

Des entreprises innovantes et agiles comme [StitchFix](https://stitchfix.com), un service de boîte qui expédie des vêtements aux consommateurs, s'appuient fortement sur l'apprentissage automatique pour les recommandations et la gestion des stocks. En fait, leurs équipes de stylisme collaborent avec leurs équipes de merchandising : "l'un de nos data scientists a expérimenté un algorithme génétique et l'a appliqué à l'habillement pour prédire quel serait un vêtement réussi qui n'existe pas aujourd'hui. Nous avons présenté cela à l'équipe de merchandising et maintenant ils peuvent l'utiliser comme un outil."
[Référence](https://www.zdnet.com/article/how-stitch-fix-uses-machine-learning-to-master-the-science-of-styling/)

## 🏥 Santé

Le secteur de la santé peut tirer parti de l'apprentissage automatique pour optimiser les tâches de recherche et également résoudre des problèmes logistiques tels que la réadmission des patients ou l'arrêt de la propagation des maladies.

### Gestion des essais cliniques

La toxicité dans les essais cliniques est une préoccupation majeure pour les fabricants de médicaments. Quelle quantité de toxicité est tolérable ? Dans cette étude, l'analyse de diverses méthodes d'essais cliniques a conduit au développement d'une nouvelle approche pour prédire les résultats des essais cliniques. Plus précisément, ils ont pu utiliser des forêts aléatoires pour produire un [classificateur](../../4-Classification/README.md) capable de distinguer entre des groupes de médicaments.
[Référence](https://www.sciencedirect.com/science/article/pii/S2451945616302914)

### Gestion des réadmissions hospitalières

Les soins hospitaliers sont coûteux, surtout lorsque les patients doivent être réadmis. Cet article discute d'une entreprise qui utilise l'apprentissage automatique pour prédire le potentiel de réadmission en utilisant des algorithmes de [clustering](../../5-Clustering/README.md). Ces clusters aident les analystes à "découvrir des groupes de réadmissions qui peuvent partager une cause commune".
[Référence](https://healthmanagement.org/c/healthmanagement/issuearticle/hospital-readmissions-and-machine-learning)

### Gestion des maladies

La récente pandémie a mis en lumière les façons dont l'apprentissage automatique peut aider à stopper la propagation des maladies. Dans cet article, vous reconnaîtrez l'utilisation de l'ARIMA, des courbes logistiques, de la régression linéaire et de la SARIMA. "Ce travail est une tentative de calculer le taux de propagation de ce virus et ainsi de prédire les décès, les rétablissements et les cas confirmés, afin de nous aider à mieux nous préparer et à survivre."
[Référence](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7979218/)

## 🌲 Écologie et technologie verte

La nature et l'écologie consistent en de nombreux systèmes sensibles où l'interaction entre les animaux et la nature est mise en avant. Il est important de pouvoir mesurer ces systèmes avec précision et d'agir de manière appropriée si quelque chose se produit, comme un incendie de forêt ou une diminution de la population animale.

### Gestion des forêts

Vous avez appris à propos de [l'apprentissage par renforcement](../../8-Reinforcement/README.md) dans les leçons précédentes. Cela peut être très utile pour essayer de prédire des motifs dans la nature. En particulier, cela peut être utilisé pour suivre des problèmes écologiques tels que les incendies de forêt et la propagation d'espèces envahissantes. Au Canada, un groupe de chercheurs a utilisé l'apprentissage par renforcement pour construire des modèles de dynamique des incendies de forêt à partir d'images satellites. En utilisant un processus innovant de "propagation spatiale (SSP)", ils ont envisagé un incendie de forêt comme "l'agent à n'importe quelle cellule du paysage". "L'ensemble des actions que le feu peut prendre à partir d'un emplacement à un moment donné inclut la propagation vers le nord, le sud, l'est ou l'ouest ou ne pas se propager."

Cette approche inverse la configuration habituelle de l'apprentissage par renforcement puisque la dynamique du processus de décision de Markov (MDP) correspondant est une fonction connue pour la propagation immédiate des incendies de forêt." Lisez-en plus sur les algorithmes classiques utilisés par ce groupe à l'adresse ci-dessous.
[Référence](https://www.frontiersin.org/articles/10.3389/fict.2018.00006/full)

### Détection de mouvements des animaux

Bien que l'apprentissage profond ait créé une révolution dans le suivi visuel des mouvements des animaux (vous pouvez construire votre propre [suiveur d'ours polaire](https://docs.microsoft.com/learn/modules/build-ml-model-with-azure-stream-analytics/?WT.mc_id=academic-77952-leestott) ici), l'apprentissage automatique classique a toujours sa place dans cette tâche.

Les capteurs pour suivre les mouvements des animaux de ferme et l'IoT utilisent ce type de traitement visuel, mais des techniques d'apprentissage automatique plus basiques sont utiles pour prétraiter les données. Par exemple, dans cet article, les postures des moutons ont été surveillées et analysées en utilisant divers algorithmes de classification. Vous pourriez reconnaître la courbe ROC à la page 335.
[Référence](https://druckhaus-hofmann.de/gallery/31-wj-feb-2020.pdf)

### ⚡️ Gestion de l'énergie

Dans nos leçons sur [la prévision des séries temporelles](../../7-TimeSeries/README.md), nous avons évoqué le concept de parcmètres intelligents pour générer des revenus pour une ville en comprenant l'offre et la demande. Cet article discute en détail de la manière dont le clustering, la régression et la prévision des séries temporelles se sont combinés pour aider à prédire la consommation future d'énergie en Irlande, basée sur la comptabilisation intelligente.
[Référence](https://www-cdn.knime.com/sites/default/files/inline-images/knime_bigdata_energy_timeseries_whitepaper.pdf)

## 💼 Assurance

Le secteur de l'assurance est un autre domaine qui utilise l'apprentissage automatique pour construire et optimiser des modèles financiers et actuariels viables.

### Gestion de la volatilité

MetLife, un fournisseur d'assurance vie, est transparent sur la manière dont ils analysent et atténuent la volatilité dans leurs modèles financiers. Dans cet article, vous remarquerez des visualisations de classification binaire et ordinale. Vous découvrirez également des visualisations de prévision.
[Référence](https://investments.metlife.com/content/dam/metlifecom/us/investments/insights/research-topics/macro-strategy/pdf/MetLifeInvestmentManagement_MachineLearnedRanking_070920.pdf)

## 🎨 Arts, Culture et Littérature

Dans les arts, par exemple dans le journalisme, il existe de nombreux problèmes intéressants. La détection de fausses nouvelles est un problème majeur car il a été prouvé qu'elle influence l'opinion des gens et même renverse des démocraties. Les musées peuvent également bénéficier de l'utilisation de l'apprentissage automatique dans tout, depuis la recherche de liens entre les artefacts jusqu'à la planification des ressources.

### Détection de fausses nouvelles

La détection de fausses nouvelles est devenue un jeu du chat et de la souris dans les médias d'aujourd'hui. Dans cet article, les chercheurs suggèrent qu'un système combinant plusieurs des techniques d'apprentissage automatique que nous avons étudiées peut être testé et que le meilleur modèle peut être déployé : "Ce système est basé sur le traitement du langage naturel pour extraire des caractéristiques des données et ensuite ces caractéristiques sont utilisées pour l'entraînement de classificateurs d'apprentissage automatique tels que Naive Bayes, Support Vector Machine (SVM), Random Forest (RF), Stochastic Gradient Descent (SGD) et Régression Logistique (LR)."
[Référence](https://www.irjet.net/archives/V7/i6/IRJET-V7I6688.pdf)

Cet article montre comment la combinaison de différents domaines de l'apprentissage automatique peut produire des résultats intéressants qui peuvent aider à arrêter la propagation de fausses nouvelles et à créer des dommages réels ; dans ce cas, l'incitation était la propagation de rumeurs sur les traitements COVID qui incitaient à la violence de masse.

### ML dans les musées

Les musées sont à l'aube d'une révolution de l'IA où le catalogage et la numérisation des collections et la recherche de liens entre les artefacts deviennent plus faciles à mesure que la technologie progresse. Des projets tels que [In Codice Ratio](https://www.sciencedirect.com/science/article/abs/pii/S0306457321001035#:~:text=1.,studies%20over%20large%20historical%20sources.) aident à déverrouiller les mystères de collections inaccessibles telles que les Archives du Vatican. Mais, l'aspect commercial des musées bénéficie également des modèles d'apprentissage automatique.

Par exemple, l'Art Institute of Chicago a construit des modèles pour prédire quels publics sont intéressés et quand ils assisteront aux expositions. L'objectif est de créer des expériences visiteurs individualisées et optimisées chaque fois que l'utilisateur visite le musée. "Au cours de l'exercice 2017, le modèle a prédit la fréquentation et les admissions avec une précision de 1 pour cent, déclare Andrew Simnick, vice-président senior de l'Art Institute."
[Reference](https://www.chicagobusiness.com/article/20180518/ISSUE01/180519840/art-institute-of-chicago-uses-data-to-make-exhibit-choices)

## 🏷 Marketing

### Segmentation des clients

Les stratégies marketing les plus efficaces ciblent les clients de différentes manières en fonction de divers groupes. Dans cet article, les utilisations des algorithmes de clustering sont discutées pour soutenir le marketing différencié. Le marketing différencié aide les entreprises à améliorer la reconnaissance de la marque, à atteindre plus de clients et à générer plus de revenus.
[Reference](https://ai.inqline.com/machine-learning-for-marketing-customer-segmentation/)

## 🚀 Défi

Identifiez un autre secteur qui bénéficie de certaines des techniques que vous avez apprises dans ce programme, et découvrez comment il utilise le ML.

## [Quiz post-conférence](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/50/)

## Révision & Auto-apprentissage

L'équipe de science des données de Wayfair a plusieurs vidéos intéressantes sur la manière dont elle utilise le ML dans son entreprise. Cela vaut la peine [d'y jeter un œil](https://www.youtube.com/channel/UCe2PjkQXqOuwkW1gw6Ameuw/videos) !

## Devoir

[Une chasse au trésor en ML](assignment.md)

**Avertissement** :  
Ce document a été traduit à l'aide de services de traduction automatique basés sur l'IA. Bien que nous nous efforçons d'assurer l'exactitude, veuillez noter que les traductions automatiques peuvent contenir des erreurs ou des inexactitudes. Le document original dans sa langue native doit être considéré comme la source faisant autorité. Pour des informations critiques, une traduction humaine professionnelle est recommandée. Nous ne sommes pas responsables des malentendus ou des interprétations erronées résultant de l'utilisation de cette traduction.