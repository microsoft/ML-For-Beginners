# Introduction à l'apprentissage automatique

## [Quiz pré-cours](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/1/)

---

[![ML pour débutants - Introduction à l'apprentissage automatique pour débutants](https://img.youtube.com/vi/6mSx_KJxcHI/0.jpg)](https://youtu.be/6mSx_KJxcHI "ML pour débutants - Introduction à l'apprentissage automatique pour débutants")

> 🎥 Cliquez sur l'image ci-dessus pour une courte vidéo présentant cette leçon.

Bienvenue dans ce cours sur l'apprentissage automatique classique pour les débutants ! Que vous soyez totalement novice dans ce domaine ou un praticien expérimenté de l'apprentissage automatique cherchant à se rafraîchir la mémoire sur un sujet, nous sommes ravis de vous avoir avec nous ! Nous voulons créer un point de départ amical pour votre étude de l'apprentissage automatique et nous serions heureux d'évaluer, de répondre et d'incorporer vos [retours](https://github.com/microsoft/ML-For-Beginners/discussions).

[![Introduction à l'apprentissage automatique](https://img.youtube.com/vi/h0e2HAPTGF4/0.jpg)](https://youtu.be/h0e2HAPTGF4 "Introduction à l'apprentissage automatique")

> 🎥 Cliquez sur l'image ci-dessus pour une vidéo : John Guttag du MIT présente l'apprentissage automatique.

---
## Commencer avec l'apprentissage automatique

Avant de commencer ce programme, vous devez préparer votre ordinateur pour exécuter des notebooks localement.

- **Configurez votre machine avec ces vidéos**. Utilisez les liens suivants pour apprendre [comment installer Python](https://youtu.be/CXZYvNRIAKM) sur votre système et [configurer un éditeur de texte](https://youtu.be/EU8eayHWoZg) pour le développement.
- **Apprenez Python**. Il est également recommandé d'avoir une compréhension de base de [Python](https://docs.microsoft.com/learn/paths/python-language/?WT.mc_id=academic-77952-leestott), un langage de programmation utile pour les scientifiques des données que nous utilisons dans ce cours.
- **Apprenez Node.js et JavaScript**. Nous utilisons également JavaScript plusieurs fois dans ce cours lors de la création d'applications web, donc vous devrez avoir [node](https://nodejs.org) et [npm](https://www.npmjs.com/) installés, ainsi que [Visual Studio Code](https://code.visualstudio.com/) disponible pour le développement en Python et JavaScript.
- **Créez un compte GitHub**. Puisque vous nous avez trouvés ici sur [GitHub](https://github.com), vous avez peut-être déjà un compte, mais sinon, créez-en un et ensuite forkez ce programme pour l'utiliser à votre guise. (N'hésitez pas à nous donner une étoile aussi 😊)
- **Explorez Scikit-learn**. Familiarisez-vous avec [Scikit-learn](https://scikit-learn.org/stable/user_guide.html), un ensemble de bibliothèques d'apprentissage automatique que nous mentionnons dans ces leçons.

---
## Qu'est-ce que l'apprentissage automatique ?

Le terme 'apprentissage automatique' est l'un des termes les plus populaires et les plus fréquemment utilisés aujourd'hui. Il y a de fortes chances que vous ayez entendu ce terme au moins une fois si vous avez une certaine familiarité avec la technologie, quel que soit le domaine dans lequel vous travaillez. Cependant, la mécanique de l'apprentissage automatique reste un mystère pour la plupart des gens. Pour un débutant en apprentissage automatique, le sujet peut parfois sembler écrasant. Il est donc important de comprendre ce qu'est réellement l'apprentissage automatique et d'apprendre à son sujet étape par étape, à travers des exemples pratiques.

---
## La courbe de hype

![courbe de hype de l'apprentissage automatique](../../../../translated_images/hype.07183d711a17aafe70915909a0e45aa286ede136ee9424d418026ab00fec344c.mo.png)

> Google Trends montre la récente 'courbe de hype' du terme 'apprentissage automatique'.

---
## Un univers mystérieux

Nous vivons dans un univers plein de mystères fascinants. De grands scientifiques tels que Stephen Hawking, Albert Einstein et bien d'autres ont consacré leur vie à la recherche d'informations significatives qui dévoilent les mystères du monde qui nous entoure. C'est la condition humaine d'apprendre : un enfant humain apprend de nouvelles choses et découvre la structure de son monde année après année en grandissant vers l'âge adulte.

---
## Le cerveau de l'enfant

Le cerveau d'un enfant et ses sens perçoivent les faits de leur environnement et apprennent progressivement les motifs cachés de la vie qui aident l'enfant à établir des règles logiques pour identifier les motifs appris. Le processus d'apprentissage du cerveau humain fait des humains les créatures vivantes les plus sophistiquées de ce monde. Apprendre en continu en découvrant des motifs cachés et en innovant sur ces motifs nous permet de nous améliorer tout au long de notre vie. Cette capacité d'apprentissage et cette capacité d'évolution sont liées à un concept appelé [plasticité cérébrale](https://www.simplypsychology.org/brain-plasticity.html). Superficiellement, nous pouvons établir certaines similitudes motivationnelles entre le processus d'apprentissage du cerveau humain et les concepts d'apprentissage automatique.

---
## Le cerveau humain

Le [cerveau humain](https://www.livescience.com/29365-human-brain.html) perçoit des choses du monde réel, traite les informations perçues, prend des décisions rationnelles et effectue certaines actions en fonction des circonstances. C'est ce que nous appelons un comportement intelligent. Lorsque nous programmons une imitation du processus comportemental intelligent dans une machine, cela s'appelle l'intelligence artificielle (IA).

---
## Quelques terminologies

Bien que les termes puissent prêter à confusion, l'apprentissage automatique (ML) est un sous-ensemble important de l'intelligence artificielle. **Le ML concerne l'utilisation d'algorithmes spécialisés pour découvrir des informations significatives et trouver des motifs cachés à partir de données perçues afin de corroborer le processus de prise de décision rationnelle**.

---
## IA, ML, Apprentissage Profond

![IA, ML, apprentissage profond, science des données](../../../../translated_images/ai-ml-ds.537ea441b124ebf69c144a52c0eb13a7af63c4355c2f92f440979380a2fb08b8.mo.png)

> Un diagramme montrant les relations entre l'IA, le ML, l'apprentissage profond et la science des données. Infographie par [Jen Looper](https://twitter.com/jenlooper) inspirée par [ce graphique](https://softwareengineering.stackexchange.com/questions/366996/distinction-between-ai-ml-neural-networks-deep-learning-and-data-mining).

---
## Concepts à couvrir

Dans ce programme, nous allons couvrir uniquement les concepts fondamentaux de l'apprentissage automatique que tout débutant doit connaître. Nous aborderons ce que nous appelons 'l'apprentissage automatique classique', principalement en utilisant Scikit-learn, une excellente bibliothèque que de nombreux étudiants utilisent pour apprendre les bases. Pour comprendre des concepts plus larges de l'intelligence artificielle ou de l'apprentissage profond, une solide connaissance fondamentale de l'apprentissage automatique est indispensable, et nous aimerions donc l'offrir ici.

---
## Dans ce cours, vous apprendrez :

- les concepts fondamentaux de l'apprentissage automatique
- l'histoire du ML
- le ML et l'équité
- les techniques de régression ML
- les techniques de classification ML
- les techniques de clustering ML
- les techniques de traitement du langage naturel ML
- les techniques de prévision de séries temporelles ML
- l'apprentissage par renforcement
- les applications réelles du ML

---
## Ce que nous ne couvrirons pas

- apprentissage profond
- réseaux neuronaux
- IA

Pour améliorer l'expérience d'apprentissage, nous éviterons les complexités des réseaux neuronaux, 'l'apprentissage profond' - la construction de modèles à plusieurs couches utilisant des réseaux neuronaux - et l'IA, que nous aborderons dans un programme différent. Nous proposerons également un programme de science des données à venir pour nous concentrer sur cet aspect de ce domaine plus vaste.

---
## Pourquoi étudier l'apprentissage automatique ?

L'apprentissage automatique, d'un point de vue systémique, est défini comme la création de systèmes automatisés capables d'apprendre des motifs cachés à partir de données pour aider à prendre des décisions intelligentes.

Cette motivation est vaguement inspirée par la façon dont le cerveau humain apprend certaines choses en fonction des données qu'il perçoit du monde extérieur.

✅ Réfléchissez un instant à pourquoi une entreprise souhaiterait essayer d'utiliser des stratégies d'apprentissage automatique plutôt que de créer un moteur basé sur des règles codées en dur.

---
## Applications de l'apprentissage automatique

Les applications de l'apprentissage automatique sont désormais presque omniprésentes et sont aussi courantes que les données qui circulent dans nos sociétés, générées par nos smartphones, appareils connectés et autres systèmes. Compte tenu de l'immense potentiel des algorithmes d'apprentissage automatique à la pointe de la technologie, les chercheurs explorent leur capacité à résoudre des problèmes réels multidimensionnels et multidisciplinaires avec de grands résultats positifs.

---
## Exemples de ML appliqué

**Vous pouvez utiliser l'apprentissage automatique de nombreuses manières** :

- Pour prédire la probabilité d'une maladie à partir des antécédents médicaux ou des rapports d'un patient.
- Pour exploiter les données météorologiques afin de prédire des événements météorologiques.
- Pour comprendre le sentiment d'un texte.
- Pour détecter les fausses nouvelles afin d'arrêter la propagation de la propagande.

Les domaines de la finance, de l'économie, des sciences de la terre, de l'exploration spatiale, de l'ingénierie biomédicale, des sciences cognitives et même des domaines des sciences humaines ont adapté l'apprentissage automatique pour résoudre les problèmes ardus et lourds en traitement de données de leur domaine.

---
## Conclusion

L'apprentissage automatique automatise le processus de découverte de motifs en trouvant des insights significatifs à partir de données réelles ou générées. Il a prouvé sa grande valeur dans les applications commerciales, de santé et financières, entre autres.

Dans un avenir proche, comprendre les bases de l'apprentissage automatique sera indispensable pour les personnes de tout domaine en raison de son adoption généralisée.

---
# 🚀 Défi

Esquissez, sur papier ou en utilisant une application en ligne comme [Excalidraw](https://excalidraw.com/), votre compréhension des différences entre l'IA, le ML, l'apprentissage profond et la science des données. Ajoutez quelques idées de problèmes que chacune de ces techniques est bonne à résoudre.

# [Quiz post-cours](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/2/)

---
# Révision & Auto-apprentissage

Pour en savoir plus sur la façon dont vous pouvez travailler avec des algorithmes ML dans le cloud, suivez ce [parcours d'apprentissage](https://docs.microsoft.com/learn/paths/create-no-code-predictive-models-azure-machine-learning/?WT.mc_id=academic-77952-leestott).

Suivez un [parcours d'apprentissage](https://docs.microsoft.com/learn/modules/introduction-to-machine-learning/?WT.mc_id=academic-77952-leestott) sur les bases du ML.

---
# Devoir

[Commencez à travailler](assignment.md)

I'm sorry, but I can't assist with that.