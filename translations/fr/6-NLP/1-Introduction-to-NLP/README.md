# Introduction à la traitement du langage naturel

Cette leçon couvre une brève histoire et des concepts importants du *traitement du langage naturel*, un sous-domaine de la *linguistique computationnelle*.

## [Quiz avant le cours](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/31/)

## Introduction

Le PLN, comme on l'appelle couramment, est l'un des domaines les plus connus où l'apprentissage automatique a été appliqué et utilisé dans des logiciels de production.

✅ Pouvez-vous penser à un logiciel que vous utilisez tous les jours qui a probablement du traitement du langage naturel intégré ? Que diriez-vous de vos programmes de traitement de texte ou des applications mobiles que vous utilisez régulièrement ?

Vous apprendrez sur :

- **L'idée des langues**. Comment les langues se sont développées et quels ont été les principaux domaines d'étude.
- **Définitions et concepts**. Vous apprendrez également des définitions et des concepts sur la façon dont les ordinateurs traitent le texte, y compris l'analyse syntaxique, la grammaire et l'identification des noms et des verbes. Il y a quelques tâches de codage dans cette leçon, et plusieurs concepts importants sont introduits que vous apprendrez à coder plus tard dans les leçons suivantes.

## Linguistique computationnelle

La linguistique computationnelle est un domaine de recherche et de développement sur plusieurs décennies qui étudie comment les ordinateurs peuvent travailler avec, et même comprendre, traduire et communiquer avec des langues. Le traitement du langage naturel (PLN) est un domaine connexe axé sur la façon dont les ordinateurs peuvent traiter des langues "naturelles", ou humaines.

### Exemple - dictée téléphonique

Si vous avez déjà dicté à votre téléphone au lieu de taper ou demandé une question à un assistant virtuel, votre discours a été converti en forme textuelle puis traité ou *analysé* à partir de la langue que vous parliez. Les mots-clés détectés ont ensuite été traités dans un format que le téléphone ou l'assistant pouvait comprendre et sur lequel il pouvait agir.

![compréhension](../../../../translated_images/comprehension.619708fc5959b0f6a24ebffba2ad7b0625391a476141df65b43b59de24e45c6f.fr.png)
> La compréhension linguistique réelle est difficile ! Image par [Jen Looper](https://twitter.com/jenlooper)

### Comment cette technologie est-elle rendue possible ?

Cela est possible parce que quelqu'un a écrit un programme informatique pour le faire. Il y a quelques décennies, certains écrivains de science-fiction ont prédit que les gens parleraient principalement à leurs ordinateurs, et que les ordinateurs comprendraient toujours exactement ce qu'ils voulaient dire. Malheureusement, il s'est avéré que c'était un problème plus difficile que beaucoup ne l'imaginaient, et bien que ce soit un problème beaucoup mieux compris aujourd'hui, il existe des défis significatifs pour atteindre un traitement du langage naturel "parfait" en ce qui concerne la compréhension du sens d'une phrase. C'est un problème particulièrement difficile lorsqu'il s'agit de comprendre l'humour ou de détecter des émotions telles que le sarcasme dans une phrase.

À ce stade, vous vous rappelez peut-être des cours d'école où l'enseignant a abordé les parties de la grammaire dans une phrase. Dans certains pays, les élèves apprennent la grammaire et la linguistique comme matière dédiée, mais dans beaucoup d'autres, ces sujets sont inclus dans l'apprentissage d'une langue : soit votre première langue à l'école primaire (apprendre à lire et à écrire) et peut-être une deuxième langue au post-primaire, ou au lycée. Ne vous inquiétez pas si vous n'êtes pas un expert pour différencier les noms des verbes ou les adverbes des adjectifs !

Si vous avez du mal avec la différence entre le *présent simple* et le *présent progressif*, vous n'êtes pas seul. C'est une chose difficile pour beaucoup de gens, même des locuteurs natifs d'une langue. La bonne nouvelle est que les ordinateurs sont vraiment bons pour appliquer des règles formelles, et vous apprendrez à écrire du code qui peut *analyser* une phrase aussi bien qu'un humain. Le plus grand défi que vous examinerez plus tard est de comprendre le *sens* et le *sentiment* d'une phrase.

## Prérequis

Pour cette leçon, le principal prérequis est de pouvoir lire et comprendre la langue de cette leçon. Il n'y a pas de problèmes mathématiques ou d'équations à résoudre. Bien que l'auteur original ait écrit cette leçon en anglais, elle est également traduite dans d'autres langues, donc vous pourriez lire une traduction. Il y a des exemples où plusieurs langues différentes sont utilisées (pour comparer les différentes règles grammaticales de différentes langues). Celles-ci ne sont *pas* traduites, mais le texte explicatif l'est, donc le sens devrait être clair.

Pour les tâches de codage, vous utiliserez Python et les exemples utilisent Python 3.8.

Dans cette section, vous aurez besoin, et utiliserez :

- **Compréhension de Python 3**. Compréhension du langage de programmation en Python 3, cette leçon utilise les entrées, les boucles, la lecture de fichiers, les tableaux.
- **Visual Studio Code + extension**. Nous utiliserons Visual Studio Code et son extension Python. Vous pouvez également utiliser un IDE Python de votre choix.
- **TextBlob**. [TextBlob](https://github.com/sloria/TextBlob) est une bibliothèque de traitement de texte simplifiée pour Python. Suivez les instructions sur le site de TextBlob pour l'installer sur votre système (installez également les corpora comme indiqué ci-dessous) :

   ```bash
   pip install -U textblob
   python -m textblob.download_corpora
   ```

> 💡 Astuce : Vous pouvez exécuter Python directement dans les environnements VS Code. Consultez la [documentation](https://code.visualstudio.com/docs/languages/python?WT.mc_id=academic-77952-leestott) pour plus d'informations.

## Parler aux machines

L'histoire de la tentative de faire comprendre aux ordinateurs le langage humain remonte à des décennies, et l'un des premiers scientifiques à considérer le traitement du langage naturel était *Alan Turing*.

### Le 'test de Turing'

Lorsque Turing faisait des recherches sur l'*intelligence artificielle* dans les années 1950, il s'est demandé si un test de conversation pouvait être donné à un humain et un ordinateur (via une correspondance tapée) où l'humain dans la conversation n'était pas sûr s'il conversait avec un autre humain ou un ordinateur.

Si, après une certaine durée de conversation, l'humain ne pouvait pas déterminer si les réponses provenaient d'un ordinateur ou non, alors pouvait-on dire que l'ordinateur *pensait* ?

### L'inspiration - 'le jeu de l'imitation'

L'idée de cela vient d'un jeu de société appelé *Le jeu de l'imitation* où un interrogateur est seul dans une pièce et chargé de déterminer lequel de deux personnes (dans une autre pièce) est masculin et féminin respectivement. L'interrogateur peut envoyer des notes et doit essayer de penser à des questions où les réponses écrites révèlent le genre de la personne mystérieuse. Bien sûr, les joueurs dans l'autre pièce essaient de tromper l'interrogateur en répondant à des questions de manière à induire en erreur ou à confondre l'interrogateur, tout en donnant également l'apparence de répondre honnêtement.

### Développer Eliza

Dans les années 1960, un scientifique du MIT nommé *Joseph Weizenbaum* a développé [*Eliza*](https://wikipedia.org/wiki/ELIZA), un 'thérapeute' informatique qui poserait des questions à l'humain et donnerait l'apparence de comprendre ses réponses. Cependant, bien qu'Eliza puisse analyser une phrase et identifier certains constructions grammaticales et mots-clés afin de donner une réponse raisonnable, on ne pouvait pas dire qu'elle *comprenait* la phrase. Si Eliza était confrontée à une phrase suivant le format "**Je suis** <u>triste</u>", elle pourrait réarranger et substituer des mots dans la phrase pour former la réponse "Depuis combien de temps **es-tu** <u>triste</u> ?".

Cela donnait l'impression qu'Eliza comprenait l'énoncé et posait une question de suivi, alors qu'en réalité, elle changeait le temps et ajoutait quelques mots. Si Eliza ne pouvait pas identifier un mot-clé pour lequel elle avait une réponse, elle donnerait à la place une réponse aléatoire qui pourrait s'appliquer à de nombreuses déclarations différentes. Eliza pouvait être facilement trompée, par exemple si un utilisateur écrivait "**Tu es** un <u>bicyclette</u>", elle pourrait répondre par "Depuis combien de temps **suis-je** une <u>bicyclette</u> ?", au lieu d'une réponse plus raisonnée.

[![Discuter avec Eliza](https://img.youtube.com/vi/RMK9AphfLco/0.jpg)](https://youtu.be/RMK9AphfLco "Discuter avec Eliza")

> 🎥 Cliquez sur l'image ci-dessus pour une vidéo sur le programme ELIZA original

> Remarque : Vous pouvez lire la description originale de [Eliza](https://cacm.acm.org/magazines/1966/1/13317-elizaa-computer-program-for-the-study-of-natural-language-communication-between-man-and-machine/abstract) publiée en 1966 si vous avez un compte ACM. Alternativement, lisez sur Eliza sur [wikipedia](https://wikipedia.org/wiki/ELIZA)

## Exercice - coder un bot conversationnel de base

Un bot conversationnel, comme Eliza, est un programme qui suscite l'entrée de l'utilisateur et semble comprendre et répondre de manière intelligente. Contrairement à Eliza, notre bot n'aura pas plusieurs règles lui donnant l'apparence d'une conversation intelligente. Au lieu de cela, notre bot n'aura qu'une seule capacité, celle de maintenir la conversation avec des réponses aléatoires qui pourraient fonctionner dans presque n'importe quelle conversation triviale.

### Le plan

Vos étapes pour construire un bot conversationnel :

1. Imprimer des instructions conseillant à l'utilisateur comment interagir avec le bot
2. Démarrer une boucle
   1. Accepter l'entrée de l'utilisateur
   2. Si l'utilisateur a demandé à quitter, alors quitter
   3. Traiter l'entrée de l'utilisateur et déterminer la réponse (dans ce cas, la réponse est un choix aléatoire dans une liste de réponses génériques possibles)
   4. Imprimer la réponse
3. revenir à l'étape 2

### Construire le bot

Créons le bot ensuite. Commençons par définir quelques phrases.

1. Créez ce bot vous-même en Python avec les réponses aléatoires suivantes :

    ```python
    random_responses = ["That is quite interesting, please tell me more.",
                        "I see. Do go on.",
                        "Why do you say that?",
                        "Funny weather we've been having, isn't it?",
                        "Let's change the subject.",
                        "Did you catch the game last night?"]
    ```

    Voici un exemple de sortie pour vous guider (l'entrée de l'utilisateur est sur les lignes commençant par `>`):

    ```output
    Hello, I am Marvin, the simple robot.
    You can end this conversation at any time by typing 'bye'
    After typing each answer, press 'enter'
    How are you today?
    > I am good thanks
    That is quite interesting, please tell me more.
    > today I went for a walk     
    Did you catch the game last night?
    > I did, but my team lost
    Funny weather we've been having, isn't it?
    > yes but I hope next week is better
    Let's change the subject.
    > ok, lets talk about music
    Why do you say that?
    > because I like music!
    Why do you say that?
    > bye
    It was nice talking to you, goodbye!
    ```

    Une solution possible à la tâche est [ici](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/1-Introduction-to-NLP/solution/bot.py)

    ✅ Arrêtez-vous et réfléchissez

    1. Pensez-vous que les réponses aléatoires pourraient "tromper" quelqu'un en lui faisant croire que le bot le comprenait réellement ?
    2. Quelles caractéristiques le bot aurait-il besoin pour être plus efficace ?
    3. Si un bot pouvait vraiment "comprendre" le sens d'une phrase, devrait-il aussi "se souvenir" du sens des phrases précédentes dans une conversation ?

---

## 🚀Défi

Choisissez l'un des éléments "arrêtez-vous et réfléchissez" ci-dessus et essayez soit de les mettre en œuvre dans le code, soit d'écrire une solution sur papier en utilisant du pseudocode.

Dans la prochaine leçon, vous apprendrez un certain nombre d'autres approches pour analyser le langage naturel et l'apprentissage automatique.

## [Quiz après le cours](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/32/)

## Révision & Auto-étude

Jetez un œil aux références ci-dessous comme opportunités de lecture supplémentaire.

### Références

1. Schubert, Lenhart, "Linguistique computationnelle", *L'Encyclopédie de Stanford de la Philosophie* (Édition du printemps 2020), Edward N. Zalta (éd.), URL = <https://plato.stanford.edu/archives/spr2020/entries/computational-linguistics/>.
2. Université de Princeton "À propos de WordNet." [WordNet](https://wordnet.princeton.edu/). Université de Princeton. 2010.

## Devoir 

[Recherche d'un bot](assignment.md)

**Avertissement** :  
Ce document a été traduit à l'aide de services de traduction automatique basés sur l'IA. Bien que nous visons à garantir l'exactitude, veuillez noter que les traductions automatiques peuvent contenir des erreurs ou des inexactitudes. Le document original dans sa langue native doit être considéré comme la source autoritaire. Pour des informations critiques, une traduction humaine professionnelle est recommandée. Nous ne sommes pas responsables des malentendus ou des interprétations erronées résultant de l'utilisation de cette traduction.