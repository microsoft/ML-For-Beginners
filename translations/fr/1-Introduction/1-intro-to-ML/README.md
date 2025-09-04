<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "73e9a7245aa57f00cd413ffd22c0ccb6",
  "translation_date": "2025-09-03T23:35:38+00:00",
  "source_file": "1-Introduction/1-intro-to-ML/README.md",
  "language_code": "fr"
}
-->
# Introduction au machine learning

## [Quiz avant le cours](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/1/)

---

[![ML pour débutants - Introduction au Machine Learning pour débutants](https://img.youtube.com/vi/6mSx_KJxcHI/0.jpg)](https://youtu.be/6mSx_KJxcHI "ML pour débutants - Introduction au Machine Learning pour débutants")

> 🎥 Cliquez sur l'image ci-dessus pour une courte vidéo expliquant cette leçon.

Bienvenue dans ce cours sur le machine learning classique pour débutants ! Que vous soyez complètement novice sur ce sujet ou un praticien expérimenté cherchant à réviser une thématique, nous sommes ravis de vous accueillir ! Nous souhaitons créer un point de départ convivial pour vos études en machine learning et serions heureux d'évaluer, répondre et intégrer vos [retours](https://github.com/microsoft/ML-For-Beginners/discussions).

[![Introduction au ML](https://img.youtube.com/vi/h0e2HAPTGF4/0.jpg)](https://youtu.be/h0e2HAPTGF4 "Introduction au ML")

> 🎥 Cliquez sur l'image ci-dessus pour une vidéo : John Guttag du MIT introduit le machine learning.

---
## Premiers pas avec le machine learning

Avant de commencer ce programme, vous devez configurer votre ordinateur pour exécuter des notebooks en local.

- **Configurez votre machine avec ces vidéos**. Utilisez les liens suivants pour apprendre [comment installer Python](https://youtu.be/CXZYvNRIAKM) sur votre système et [configurer un éditeur de texte](https://youtu.be/EU8eayHWoZg) pour le développement.
- **Apprenez Python**. Il est également recommandé d'avoir une compréhension de base de [Python](https://docs.microsoft.com/learn/paths/python-language/?WT.mc_id=academic-77952-leestott), un langage de programmation utile pour les data scientists que nous utilisons dans ce cours.
- **Apprenez Node.js et JavaScript**. Nous utilisons également JavaScript à quelques reprises dans ce cours pour créer des applications web, donc vous devrez avoir [node](https://nodejs.org) et [npm](https://www.npmjs.com/) installés, ainsi que [Visual Studio Code](https://code.visualstudio.com/) disponible pour le développement en Python et JavaScript.
- **Créez un compte GitHub**. Puisque vous nous avez trouvés ici sur [GitHub](https://github.com), vous avez peut-être déjà un compte, mais sinon, créez-en un, puis forkez ce programme pour l'utiliser vous-même. (N'hésitez pas à nous donner une étoile aussi 😊)
- **Explorez Scikit-learn**. Familiarisez-vous avec [Scikit-learn](https://scikit-learn.org/stable/user_guide.html), un ensemble de bibliothèques ML que nous utilisons dans ces leçons.

---
## Qu'est-ce que le machine learning ?

Le terme "machine learning" est l'un des termes les plus populaires et fréquemment utilisés aujourd'hui. Il est fort probable que vous ayez entendu ce terme au moins une fois si vous avez une certaine familiarité avec la technologie, quel que soit votre domaine. Cependant, les mécanismes du machine learning restent un mystère pour la plupart des gens. Pour un débutant, le sujet peut parfois sembler écrasant. Il est donc important de comprendre ce qu'est réellement le machine learning et d'apprendre à son sujet étape par étape, à travers des exemples pratiques.

---
## La courbe de la hype

![courbe de la hype ML](../../../../translated_images/hype.07183d711a17aafe70915909a0e45aa286ede136ee9424d418026ab00fec344c.fr.png)

> Google Trends montre la récente "courbe de la hype" du terme "machine learning"

---
## Un univers mystérieux

Nous vivons dans un univers rempli de mystères fascinants. De grands scientifiques comme Stephen Hawking, Albert Einstein, et bien d'autres ont consacré leur vie à rechercher des informations significatives pour dévoiler les mystères du monde qui nous entoure. C'est la condition humaine d'apprentissage : un enfant humain apprend de nouvelles choses et découvre la structure de son monde année après année en grandissant.

---
## Le cerveau de l'enfant

Le cerveau et les sens d'un enfant perçoivent les faits de leur environnement et apprennent progressivement les schémas cachés de la vie, ce qui aide l'enfant à élaborer des règles logiques pour identifier ces schémas. Le processus d'apprentissage du cerveau humain fait de nous les créatures vivantes les plus sophistiquées de ce monde. Apprendre continuellement en découvrant des schémas cachés, puis en innovant à partir de ces schémas, nous permet de nous améliorer tout au long de notre vie. Cette capacité d'apprentissage et d'évolution est liée à un concept appelé [plasticité cérébrale](https://www.simplypsychology.org/brain-plasticity.html). Superficiellement, nous pouvons établir des similitudes motivantes entre le processus d'apprentissage du cerveau humain et les concepts du machine learning.

---
## Le cerveau humain

Le [cerveau humain](https://www.livescience.com/29365-human-brain.html) perçoit des éléments du monde réel, traite les informations perçues, prend des décisions rationnelles et effectue certaines actions en fonction des circonstances. C'est ce que nous appelons se comporter intelligemment. Lorsque nous programmons une imitation de ce processus comportemental intelligent dans une machine, cela s'appelle l'intelligence artificielle (IA).

---
## Quelques terminologies

Bien que les termes puissent prêter à confusion, le machine learning (ML) est un sous-ensemble important de l'intelligence artificielle. **Le ML consiste à utiliser des algorithmes spécialisés pour découvrir des informations significatives et trouver des schémas cachés à partir des données perçues afin de corroborer le processus de prise de décision rationnelle**.

---
## IA, ML, Deep Learning

![IA, ML, deep learning, data science](../../../../translated_images/ai-ml-ds.537ea441b124ebf69c144a52c0eb13a7af63c4355c2f92f440979380a2fb08b8.fr.png)

> Un diagramme montrant les relations entre l'IA, le ML, le deep learning et la data science. Infographie par [Jen Looper](https://twitter.com/jenlooper) inspirée de [ce graphique](https://softwareengineering.stackexchange.com/questions/366996/distinction-between-ai-ml-neural-networks-deep-learning-and-data-mining)

---
## Concepts à couvrir

Dans ce programme, nous allons couvrir uniquement les concepts fondamentaux du machine learning qu'un débutant doit connaître. Nous abordons ce que nous appelons le "machine learning classique", principalement en utilisant Scikit-learn, une excellente bibliothèque que de nombreux étudiants utilisent pour apprendre les bases. Pour comprendre les concepts plus larges de l'intelligence artificielle ou du deep learning, une solide connaissance fondamentale du machine learning est indispensable, et nous souhaitons vous l'offrir ici.

---
## Dans ce cours, vous apprendrez :

- les concepts fondamentaux du machine learning
- l'histoire du ML
- le ML et l'équité
- les techniques de régression en ML
- les techniques de classification en ML
- les techniques de clustering en ML
- les techniques de traitement du langage naturel en ML
- les techniques de prévision de séries temporelles en ML
- l'apprentissage par renforcement
- les applications réelles du ML

---
## Ce que nous ne couvrirons pas

- le deep learning
- les réseaux neuronaux
- l'IA

Pour une meilleure expérience d'apprentissage, nous éviterons les complexités des réseaux neuronaux, du "deep learning" - la construction de modèles à plusieurs couches utilisant des réseaux neuronaux - et de l'IA, que nous aborderons dans un autre programme. Nous proposerons également un programme à venir sur la data science pour se concentrer sur cet aspect de ce domaine plus vaste.

---
## Pourquoi étudier le machine learning ?

Le machine learning, d'un point de vue système, est défini comme la création de systèmes automatisés capables d'apprendre des schémas cachés à partir de données pour aider à prendre des décisions intelligentes.

Cette motivation s'inspire vaguement de la manière dont le cerveau humain apprend certaines choses à partir des données qu'il perçoit du monde extérieur.

✅ Réfléchissez un instant à pourquoi une entreprise pourrait vouloir utiliser des stratégies de machine learning plutôt que de créer un moteur basé sur des règles codées en dur.

---
## Applications du machine learning

Les applications du machine learning sont désormais presque partout et sont aussi omniprésentes que les données qui circulent dans nos sociétés, générées par nos smartphones, appareils connectés et autres systèmes. Compte tenu du potentiel immense des algorithmes de machine learning de pointe, les chercheurs explorent leurs capacités à résoudre des problèmes réels multidimensionnels et multidisciplinaires avec des résultats très positifs.

---
## Exemples de ML appliqué

**Vous pouvez utiliser le machine learning de nombreuses façons** :

- Pour prédire la probabilité d'une maladie à partir des antécédents médicaux ou des rapports d'un patient.
- Pour exploiter les données météorologiques afin de prévoir des événements climatiques.
- Pour comprendre le sentiment d'un texte.
- Pour détecter les fausses nouvelles et stopper la propagation de la propagande.

La finance, l'économie, les sciences de la Terre, l'exploration spatiale, le génie biomédical, les sciences cognitives, et même les domaines des sciences humaines ont adopté le machine learning pour résoudre les problèmes complexes et gourmands en traitement de données de leur domaine.

---
## Conclusion

Le machine learning automatise le processus de découverte de schémas en trouvant des informations significatives à partir de données réelles ou générées. Il s'est avéré extrêmement précieux dans les applications commerciales, de santé et financières, entre autres.

Dans un avenir proche, comprendre les bases du machine learning deviendra indispensable pour les personnes de tous domaines en raison de son adoption généralisée.

---
# 🚀 Défi

Dessinez, sur papier ou en utilisant une application en ligne comme [Excalidraw](https://excalidraw.com/), votre compréhension des différences entre l'IA, le ML, le deep learning et la data science. Ajoutez quelques idées de problèmes que chacune de ces techniques est bonne à résoudre.

# [Quiz après le cours](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/2/)

---
# Révision et auto-apprentissage

Pour en savoir plus sur la façon de travailler avec des algorithmes de ML dans le cloud, suivez ce [parcours d'apprentissage](https://docs.microsoft.com/learn/paths/create-no-code-predictive-models-azure-machine-learning/?WT.mc_id=academic-77952-leestott).

Suivez un [parcours d'apprentissage](https://docs.microsoft.com/learn/modules/introduction-to-machine-learning/?WT.mc_id=academic-77952-leestott) sur les bases du ML.

---
# Devoir

[Mettez-vous en route](assignment.md)

---

**Avertissement** :  
Ce document a été traduit à l'aide du service de traduction automatique [Co-op Translator](https://github.com/Azure/co-op-translator). Bien que nous nous efforcions d'assurer l'exactitude, veuillez noter que les traductions automatisées peuvent contenir des erreurs ou des inexactitudes. Le document original dans sa langue d'origine doit être considéré comme la source faisant autorité. Pour des informations critiques, il est recommandé de recourir à une traduction professionnelle réalisée par un humain. Nous déclinons toute responsabilité en cas de malentendus ou d'interprétations erronées résultant de l'utilisation de cette traduction.