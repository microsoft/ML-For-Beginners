<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "1c2ec40cf55c98a028a359c27ef7e45a",
  "translation_date": "2025-09-04T23:07:49+00:00",
  "source_file": "6-NLP/1-Introduction-to-NLP/README.md",
  "language_code": "fr"
}
-->
# Introduction au traitement du langage naturel

Cette leçon couvre une brève histoire et les concepts importants du *traitement du langage naturel*, un sous-domaine de la *linguistique computationnelle*.

## [Quiz avant la leçon](https://ff-quizzes.netlify.app/en/ml/)

## Introduction

Le traitement du langage naturel, ou NLP, est l'un des domaines les plus connus où l'apprentissage automatique a été appliqué et utilisé dans des logiciels de production.

✅ Pouvez-vous penser à un logiciel que vous utilisez quotidiennement et qui intègre probablement du NLP ? Que dire des programmes de traitement de texte ou des applications mobiles que vous utilisez régulièrement ?

Vous apprendrez :

- **L'idée des langues**. Comment les langues se sont développées et quels ont été les principaux domaines d'étude.
- **Définitions et concepts**. Vous apprendrez également des définitions et des concepts sur la manière dont les ordinateurs traitent le texte, notamment l'analyse syntaxique, la grammaire et l'identification des noms et des verbes. Cette leçon inclut des tâches de codage et introduit plusieurs concepts importants que vous apprendrez à coder dans les prochaines leçons.

## Linguistique computationnelle

La linguistique computationnelle est un domaine de recherche et de développement qui, depuis plusieurs décennies, étudie comment les ordinateurs peuvent travailler avec les langues, les comprendre, les traduire et même communiquer avec elles. Le traitement du langage naturel (NLP) est un domaine connexe qui se concentre sur la manière dont les ordinateurs peuvent traiter les langues dites "naturelles", c'est-à-dire humaines.

### Exemple - dictée sur téléphone

Si vous avez déjà dicté à votre téléphone au lieu de taper ou posé une question à un assistant virtuel, votre discours a été converti en texte, puis traité ou *analysé* à partir de la langue que vous avez parlée. Les mots-clés détectés ont ensuite été transformés dans un format que le téléphone ou l'assistant pouvait comprendre et utiliser.

![comprehension](../../../../6-NLP/1-Introduction-to-NLP/images/comprehension.png)
> Comprendre réellement une langue est difficile ! Image par [Jen Looper](https://twitter.com/jenlooper)

### Comment cette technologie est-elle rendue possible ?

Cela est possible parce que quelqu'un a écrit un programme informatique pour le faire. Il y a quelques décennies, certains écrivains de science-fiction prédisaient que les gens parleraient principalement à leurs ordinateurs et que ces derniers comprendraient toujours exactement ce qu'ils voulaient dire. Malheureusement, il s'est avéré que c'était un problème plus difficile que prévu. Bien que ce problème soit mieux compris aujourd'hui, il reste des défis importants pour atteindre un traitement du langage naturel "parfait", notamment lorsqu'il s'agit de comprendre le sens d'une phrase. Cela est particulièrement difficile lorsqu'il s'agit de comprendre l'humour ou de détecter des émotions comme le sarcasme dans une phrase.

À ce stade, vous vous souvenez peut-être des cours à l'école où l'enseignant expliquait les parties de la grammaire dans une phrase. Dans certains pays, les élèves apprennent la grammaire et la linguistique comme une matière dédiée, mais dans beaucoup d'autres, ces sujets sont inclus dans l'apprentissage d'une langue : soit votre langue maternelle à l'école primaire (apprendre à lire et écrire), soit une langue étrangère au collège ou au lycée. Ne vous inquiétez pas si vous n'êtes pas expert pour différencier les noms des verbes ou les adverbes des adjectifs !

Si vous avez du mal à distinguer le *présent simple* du *présent progressif*, vous n'êtes pas seul. C'est un défi pour de nombreuses personnes, même les locuteurs natifs d'une langue. La bonne nouvelle est que les ordinateurs sont très bons pour appliquer des règles formelles, et vous apprendrez à écrire du code capable d'*analyser* une phrase aussi bien qu'un humain. Le défi plus grand que vous examinerez plus tard est de comprendre le *sens* et le *sentiment* d'une phrase.

## Prérequis

Pour cette leçon, le principal prérequis est de pouvoir lire et comprendre la langue de cette leçon. Il n'y a pas de problèmes mathématiques ou d'équations à résoudre. Bien que l'auteur original ait écrit cette leçon en anglais, elle est également traduite dans d'autres langues, donc vous pourriez lire une traduction. Il y a des exemples où plusieurs langues différentes sont utilisées (pour comparer les différentes règles grammaticales des langues). Ces exemples *ne sont pas* traduits, mais le texte explicatif l'est, donc le sens devrait être clair.

Pour les tâches de codage, vous utiliserez Python, et les exemples sont basés sur Python 3.8.

Dans cette section, vous aurez besoin et utiliserez :

- **Compréhension de Python 3**. Compréhension du langage de programmation Python 3, cette leçon utilise des entrées, des boucles, la lecture de fichiers et des tableaux.
- **Visual Studio Code + extension**. Nous utiliserons Visual Studio Code et son extension Python. Vous pouvez également utiliser un IDE Python de votre choix.
- **TextBlob**. [TextBlob](https://github.com/sloria/TextBlob) est une bibliothèque simplifiée de traitement de texte pour Python. Suivez les instructions sur le site de TextBlob pour l'installer sur votre système (installez également les corpus, comme indiqué ci-dessous) :

   ```bash
   pip install -U textblob
   python -m textblob.download_corpora
   ```

> 💡 Astuce : Vous pouvez exécuter Python directement dans les environnements VS Code. Consultez les [docs](https://code.visualstudio.com/docs/languages/python?WT.mc_id=academic-77952-leestott) pour plus d'informations.

## Parler aux machines

L'histoire de la tentative de faire comprendre aux ordinateurs le langage humain remonte à plusieurs décennies, et l'un des premiers scientifiques à envisager le traitement du langage naturel fut *Alan Turing*.

### Le 'test de Turing'

Lorsque Turing recherchait l'*intelligence artificielle* dans les années 1950, il s'est demandé si un test conversationnel pouvait être donné à un humain et à un ordinateur (via une correspondance écrite) où l'humain dans la conversation ne savait pas s'il conversait avec un autre humain ou un ordinateur.

Si, après une certaine durée de conversation, l'humain ne pouvait pas déterminer si les réponses provenaient d'un ordinateur ou non, alors pourrait-on dire que l'ordinateur *pense* ?

### L'inspiration - 'le jeu de l'imitation'

L'idée de ce test vient d'un jeu de société appelé *Le Jeu de l'Imitation*, où un interrogateur est seul dans une pièce et doit déterminer lesquels des deux individus (dans une autre pièce) sont respectivement un homme et une femme. L'interrogateur peut envoyer des notes et doit essayer de poser des questions dont les réponses écrites révèlent le genre de la personne mystère. Bien sûr, les joueurs dans l'autre pièce essaient de tromper l'interrogateur en répondant aux questions de manière à le dérouter ou le confondre, tout en donnant l'impression de répondre honnêtement.

### Développer Eliza

Dans les années 1960, un scientifique du MIT nommé *Joseph Weizenbaum* a développé [*Eliza*](https://wikipedia.org/wiki/ELIZA), un "thérapeute" informatique qui posait des questions à l'humain et donnait l'impression de comprendre ses réponses. Cependant, bien qu'Eliza puisse analyser une phrase et identifier certains éléments grammaticaux et mots-clés pour donner une réponse raisonnable, on ne pouvait pas dire qu'elle *comprenait* la phrase. Si Eliza recevait une phrase suivant le format "**Je suis** <u>triste</u>", elle pouvait réarranger et substituer des mots dans la phrase pour former la réponse "Depuis combien de temps **êtes-vous** <u>triste</u> ?".

Cela donnait l'impression qu'Eliza comprenait la déclaration et posait une question de suivi, alors qu'en réalité, elle changeait le temps et ajoutait quelques mots. Si Eliza ne pouvait pas identifier un mot-clé pour lequel elle avait une réponse, elle donnait une réponse aléatoire qui pouvait s'appliquer à de nombreuses déclarations différentes. Eliza pouvait être facilement trompée, par exemple si un utilisateur écrivait "**Vous êtes** une <u>bicyclette</u>", elle pouvait répondre "Depuis combien de temps **suis-je** une <u>bicyclette</u> ?", au lieu d'une réponse plus réfléchie.

[![Discuter avec Eliza](https://img.youtube.com/vi/RMK9AphfLco/0.jpg)](https://youtu.be/RMK9AphfLco "Discuter avec Eliza")

> 🎥 Cliquez sur l'image ci-dessus pour une vidéo sur le programme original ELIZA

> Note : Vous pouvez lire la description originale d'[Eliza](https://cacm.acm.org/magazines/1966/1/13317-elizaa-computer-program-for-the-study-of-natural-language-communication-between-man-and-machine/abstract) publiée en 1966 si vous avez un compte ACM. Sinon, lisez à propos d'Eliza sur [wikipedia](https://wikipedia.org/wiki/ELIZA).

## Exercice - coder un bot conversationnel basique

Un bot conversationnel, comme Eliza, est un programme qui sollicite des entrées utilisateur et semble comprendre et répondre intelligemment. Contrairement à Eliza, notre bot n'aura pas plusieurs règles lui donnant l'apparence d'une conversation intelligente. Au lieu de cela, notre bot aura une seule capacité : maintenir la conversation avec des réponses aléatoires qui pourraient fonctionner dans presque n'importe quelle conversation triviale.

### Le plan

Les étapes pour construire un bot conversationnel :

1. Afficher des instructions conseillant à l'utilisateur comment interagir avec le bot
2. Démarrer une boucle
   1. Accepter les entrées utilisateur
   2. Si l'utilisateur demande à quitter, alors quitter
   3. Traiter les entrées utilisateur et déterminer une réponse (dans ce cas, la réponse est un choix aléatoire parmi une liste de réponses génériques possibles)
   4. Afficher la réponse
3. Revenir à l'étape 2

### Construire le bot

Créons le bot maintenant. Nous commencerons par définir quelques phrases.

1. Créez ce bot vous-même en Python avec les réponses aléatoires suivantes :

    ```python
    random_responses = ["That is quite interesting, please tell me more.",
                        "I see. Do go on.",
                        "Why do you say that?",
                        "Funny weather we've been having, isn't it?",
                        "Let's change the subject.",
                        "Did you catch the game last night?"]
    ```

    Voici un exemple de sortie pour vous guider (les entrées utilisateur commencent par `>`):

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

    1. Pensez-vous que les réponses aléatoires pourraient "tromper" quelqu'un en lui faisant croire que le bot comprenait réellement ?
    2. Quelles fonctionnalités le bot devrait-il avoir pour être plus efficace ?
    3. Si un bot pouvait vraiment "comprendre" le sens d'une phrase, devrait-il également "se souvenir" du sens des phrases précédentes dans une conversation ?

---

## 🚀Défi

Choisissez l'un des éléments "arrêtez-vous et réfléchissez" ci-dessus et essayez soit de l'implémenter en code, soit d'écrire une solution sur papier en pseudocode.

Dans la prochaine leçon, vous apprendrez plusieurs autres approches pour analyser le langage naturel et l'apprentissage automatique.

## [Quiz après la leçon](https://ff-quizzes.netlify.app/en/ml/)

## Révision & Étude personnelle

Consultez les références ci-dessous pour des opportunités de lecture complémentaire.

### Références

1. Schubert, Lenhart, "Computational Linguistics", *The Stanford Encyclopedia of Philosophy* (Spring 2020 Edition), Edward N. Zalta (ed.), URL = <https://plato.stanford.edu/archives/spr2020/entries/computational-linguistics/>.
2. Princeton University "About WordNet." [WordNet](https://wordnet.princeton.edu/). Princeton University. 2010. 

## Devoir 

[Recherchez un bot](assignment.md)

---

**Avertissement** :  
Ce document a été traduit à l'aide du service de traduction automatique [Co-op Translator](https://github.com/Azure/co-op-translator). Bien que nous nous efforcions d'assurer l'exactitude, veuillez noter que les traductions automatisées peuvent contenir des erreurs ou des inexactitudes. Le document original dans sa langue d'origine doit être considéré comme la source faisant autorité. Pour des informations critiques, il est recommandé de recourir à une traduction professionnelle réalisée par un humain. Nous déclinons toute responsabilité en cas de malentendus ou d'interprétations erronées résultant de l'utilisation de cette traduction.