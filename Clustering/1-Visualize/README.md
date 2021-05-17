# Introduction to Clustering

[![No One Like You by PSquare](https://img.youtube.com/vi/ty2advRiWJM/0.jpg)](https://youtu.be/ty2advRiWJM "No One Like You by PSquare")

> While you're studying Machine Learning with Clustering, enjoy some Nigerian Dance Hall tracks - this is a highly rated song from 2014 by PSquare.
## [Pre-lecture quiz](link-to-quiz-app)

Clustering is a type of unsupervised learning that presumes that a dataset is unlabelled. It uses various algorithms to sort through unlabeled data and provide groupings according to patterns it discerns in the data. Clustering is very useful for data exploration. Let's see if it can help discover trends and patterns in the way Nigerian audiences consume music.

✅ Take a minute to think about the uses of clustering. In real life, clustering happens whenever you have a pile of laundry and need to sort out your family members' clothes 🧦👕👖🩲. In data science, clustering happens when trying to analyze a user's preferences, or determine the characteristics of any unlabeled dataset. Clustering, in a way, helps make sense of chaos.
### Introduction

[Scikit-Learn offers a large array](https://scikit-learn.org/stable/modules/clustering.html) of methods to perform clustering. The type you choose will depend on your use case. According to the documentation, each method has various benefits. Here is a simplified table of the methods supported by Scikit-Learn and their appropriate use cases:

| Method name                  | Use case                                                               |
| :--------------------------- | :--------------------------------------------------------------------- |
| K-Means                      | general purpose, inductive                                             |
| Affinity propagation         | many, uneven clusters, inductive                                       |
| Mean-shift                   | many, uneven clusters, inductive                                       |
| Spectral clustering          | few, even clusters, transductive                                       |
| Ward hierarchical clustering | many, constrained clusters, transductive                               |
| Agglomerative clustering     | many, constrained, non Euclidan distances, transductive                |
| DBSCAN                       | non-flat geometry, uneven clusters, transductive                       |
| OPTICS                       | non-flat geometry, uneven clusters with variable density, transductive |
| Gaussian mixtures            | flat geometry, inductive                                               |
| BIRCH                        | large dataset with outliers, inductive                                 |

> 🎓 Let's unpack some vocabulary:
>
> - 'transductive' vs. 'inductive'
>  - 'non-flat' vs. 'flat' geometry
>  - 'distances'
>  - 'constrained'
>  - 'density'
### Preparation

Open the notebook.ipynb file in this folder and append the song data

---



## 🚀Challenge

Add a challenge for students to work on collaboratively in class to enhance the project

Optional: add a screenshot of the completed lesson's UI if appropriate

## [Post-lecture quiz](link-to-quiz-app)

## Review & Self Study

**Assignment**: [Assignment Name](assignment.md)
