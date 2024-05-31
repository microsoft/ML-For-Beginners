# Introduction au machine learning

[![ML, AI, deep learning - Quelle est la différence ?](https://img.youtube.com/vi/lTd9RSxS9ZE/0.jpg)](https://youtu.be/lTd9RSxS9ZE "ML, AI, deep learning - What's the difference?")

> 🎥 Cliquer sur l'image ci-dessus afin de regarder une vidéo expliquant la différence entre machine learning, AI et deep learning.

## [Quiz préalable](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/1?loc=fr)

### Introduction

Bienvenue à ce cours sur le machine learning classique pour débutant ! Que vous soyez complètement nouveau sur ce sujet ou que vous soyez un professionnel du ML expérimenté cherchant à peaufiner vos connaissances, nous sommes heureux de vous avoir avec nous ! Nous voulons créer un tremplin chaleureux pour vos études en ML et serions ravis d'évaluer, de répondre et d'apprendre de vos retours d'[expériences](https://github.com/microsoft/ML-For-Beginners/discussions).

[![Introduction au ML](https://img.youtube.com/vi/h0e2HAPTGF4/0.jpg)](https://youtu.be/h0e2HAPTGF4 "Introduction to ML")

> 🎥 Cliquer sur l'image ci-dessus afin de regarder une vidéo: John Guttag du MIT introduit le machine learning
### Débuter avec le machine learning

Avant de commencer avec ce cours, vous aurez besoin d'un ordinateur configuré et prêt à faire tourner des notebooks (jupyter) localement.

- **Configurer votre ordinateur avec ces vidéos**. Apprendre comment configurer votre ordinateur avec cette [série de vidéos](https://www.youtube.com/playlist?list=PLlrxD0HtieHhS8VzuMCfQD4uJ9yne1mE6).
- **Apprendre Python**. Il est aussi recommandé d'avoir une connaissance basique de [Python](https://docs.microsoft.com/learn/paths/python-language/?WT.mc_id=academic-77952-leestott), un langage de programmaton utile pour les data scientist que nous utilisons tout au long de ce cours.
- **Apprendre Node.js et Javascript**. Nous utilisons aussi Javascript par moment dans ce cours afin de construire des applications WEB, vous aurez donc besoin de [node](https://nodejs.org) et [npm](https://www.npmjs.com/) installé, ainsi que de [Visual Studio Code](https://code.visualstudio.com/) pour développer en Python et Javascript.
- **Créer un compte GitHub**. Comme vous nous avez trouvé sur [GitHub](https://github.com), vous y avez sûrement un compte, mais si non, créez en un et répliquez ce cours afin de l'utiliser à votre grés. (N'oublier pas de nous donner une étoile aussi 😊)
- **Explorer Scikit-learn**. Familiariser vous avec [Scikit-learn](https://scikit-learn.org/stable/user_guide.html), un ensemble de librairies ML que nous mentionnons dans nos leçons.

### Qu'est-ce que le machine learning

Le terme `machine learning` est un des mots les plus populaire et le plus utilisé ces derniers temps. Il y a une probabilité accrue que vous l'ayez entendu au moins une fois si vous avez une appétence pour la technologie indépendamment du domaine dans lequel vous travaillez. Le fonctionnement du machine learning, cependant, reste un mystère pour la plupart des personnes. Pour un débutant en machine learning, le sujet peut nous submerger. Ainsi, il est important de comprendre ce qu'est le machine learning et de l'apprendre petit à petit au travers d'exemples pratiques.   

![ml hype curve](../images/hype.png)

> Google Trends montre la récente 'courbe de popularité' pour le mot 'machine learning'

Nous vivons dans un univers rempli de mystères fascinants. De grands scientifiques comme Stephen Hawking, Albert Einstein et pleins d'autres ont dévoués leur vie à la recherche d'informations utiles afin de dévoiler les mystères qui nous entourent. C'est la condition humaine pour apprendre : un enfant apprend de nouvelles choses et découvre la structure du monde année après année jusqu'à qu'ils deviennent adultes.

Le cerveau d'un enfant et ses sens perçoivent l'environnement qui les entourent et apprennent graduellement des schémas non observés de la vie qui vont l'aider à fabriquer des règles logiques afin d'identifier les schémas appris. Le processus d'apprentissage du cerveau humain est ce que rend les hommes comme la créature la plus sophistiquée du monde vivant. Apprendre continuellement par la découverte de schémas non observés et ensuite innover sur ces schémas nous permet de nous améliorer tout au long de notre vie. Cette capacité d'apprendre et d'évoluer est liée au concept de [plasticité neuronale](https://www.simplypsychology.org/brain-plasticity.html), nous pouvons tirer quelques motivations similaires entre le processus d'apprentissage du cerveau humain et le concept de machine learning.

Le [cerveau humain](https://www.livescience.com/29365-human-brain.html) perçoit des choses du monde réel, assimile les informations perçues, fait des décisions rationnelles et entreprend certaines actions selon le contexte. C'est ce que l'on appelle se comporter intelligemment. Lorsque nous programmons une reproduction du processus de ce comportement à une machine, c'est ce que l'on appelle intelligence artificielle (IA).

Bien que le terme puisse être confus, le machine learning (ML) est un important sous-ensemble de l'intelligence artificielle. **Le ML consiste à utiliser des algorithmes spécialisés afin de découvrir des informations utiles et de trouver des schémas non observés depuis des données perçues pour corroborer un processus de décision rationnel**.

![AI, ML, deep learning, data science](../images/ai-ml-ds.png)

> Un diagramme montrant les relations entre AI, ML, deep learning et data science. Infographie par [Jen Looper](https://twitter.com/jenlooper) et inspiré par [ce graphique](https://softwareengineering.stackexchange.com/questions/366996/distinction-between-ai-ml-neural-networks-deep-learning-and-data-mining)

## Ce que vous allez apprendre dans ce cours

Dans ce cours, nous allons nous concentrer sur les concepts clés du machine learning qu'un débutant se doit de connaître. Nous parlerons de ce que l'on appelle le 'machine learning classique' en utilisant principalement Scikit-learn, une excellente librairie que beaucoup d'étudiants utilisent afin d'apprendre les bases. Afin de comprendre les concepts plus larges de l'intelligence artificielle ou du deep learning, une profonde connaissance en machine learning est indispensable, et c'est ce que nous aimerions fournir ici.

Dans ce cours, vous allez apprendre :

- Les concepts clés du machine learning
- L'histoire du ML
- ML et équité (fairness)
- Les techniques de régression ML
- Les techniques de classification ML
- Les techniques de regroupement (clustering) ML
- Les techniques du traitement automatique des langues (NLP) ML
- Les techniques de prédictions à partir de séries chronologiques ML
- Apprentissage renforcé
- D'applications réels du ML

## Ce que nous ne couvrirons pas

- Deep learning
- Neural networks
- IA

Afin d'avoir la meilleure expérience d'apprentissage, nous éviterons les complexités des réseaux neuronaux, du 'deep learning' (construire un modèle utilisant plusieurs couches de réseaux neuronaux) et IA, dont nous parlerons dans un cours différent. Nous offirons aussi un cours à venir sur la data science pour nous concentrer sur cet aspect de champs très large.

## Pourquoi étudier le machine learning ?

Le machine learning, depuis une perspective systémique, est défini comme la création de systèmes automatiques pouvant apprendre des schémas non observés depuis des données afin d'aider à prendre des décisions intelligentes.

Ce but est faiblement inspiré de la manière dont le cerveau humain apprend certaines choses depuis les données qu'il perçoit du monde extérieur.

✅ Pensez une minute aux raisons qu'une entreprise aurait d'essayer d'utiliser des stratégies de machine learning au lieu de créer des règles codés en dur.

### Les applications du machine learning

Les applications du machine learning sont maintenant pratiquement partout, et sont aussi omniprésentes que les données qui circulent autour de notre société (générées par nos smartphones, appareils connectés ou autres systèmes). En prenant en considération l'immense potentiel des algorithmes dernier cri de machine learning, les chercheurs ont pu exploiter leurs capacités afin de résoudre des problèmes multidimensionnels et interdisciplinaires de la vie avec d'important retours positifs.

**Vous pouvez utiliser le machine learning de plusieurs manières** :

- Afin de prédire la possibilité d'avoir une maladie à partir des données médicales d'un patient.
- Pour tirer parti des données météorologiques afin de prédire les événements météorologiques.
- Afin de comprendre le sentiment d'un texte.
- Afin de détecter les fake news pour stopper la propagation de la propagande.

La finance, l'économie, les sciences de la terre, l'exploration spatiale, le génie biomédical, les sciences cognitives et même les domaines des sciences humaines ont adapté le machine learning pour résoudre les problèmes ardus et lourds de traitement des données dans leur domaine respectif.

Le machine learning automatise le processus de découverte de modèles en trouvant des informations significatives à partir de données réelles ou générées. Il s'est avéré très utile dans les applications commerciales, de santé et financières, entre autres.

Dans un avenir proche, comprendre les bases du machine learning sera indispensable pour les personnes de tous les domaines en raison de son adoption généralisée.

---
## 🚀 Challenge

Esquisser, sur papier ou à l'aide d'une application en ligne comme [Excalidraw](https://excalidraw.com/), votre compréhension des différences entre l'IA, le ML, le deep learning et la data science. Ajouter quelques idées de problèmes que chacune de ces techniques est bonne à résoudre.

## [Quiz de validation des connaissances](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/2?loc=fr)

## Révision et auto-apprentissage

Pour en savoir plus sur la façon dont vous pouvez utiliser les algorithmes de ML dans le cloud, suivez ce [Parcours d'apprentissage](https://docs.microsoft.com/learn/paths/create-no-code-predictive-models-azure-machine-learning/?WT.mc_id=academic-77952-leestott).

## Devoir

[Être opérationnel](assignment.fr.md)
