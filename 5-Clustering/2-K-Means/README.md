# K-Means Clustering

[![Andrew Ng explains Clustering](https://img.youtube.com/vi/hDmNF9JG3lo/0.jpg)](https://youtu.be/hDmNF9JG3lo "Andrew Ng explains Clustering")

> 🎥 Click the image above for a video: Andrew Ng explains Clustering

## [Pre-lecture quiz](https://jolly-sea-0a877260f.azurestaticapps.net/quiz/27/)

In this lesson, you will learn how to create clusters using Scikit-Learn and the Nigerian music dataset you imported earlier. We will cover the basics of K-Means for Clustering. Keep in mind that, as you learned in the earlier lesson, there are many ways to work with clusters and the method you use depends on your data. We will try K-Means as it's the most common Clustering technique. Let's get started!

- Data variance
- Silhouette Scoring
- Elbow Method
- K-Means for Clustering

### Introduction

### Prerequisite
### Preparation

Preparatory steps to start this lesson

### Silhouette score

"The value of the Silhouette score varies from -1 to 1. If the score is 1, the cluster is dense and well-separated than other clusters. A value near 0 represents overlapping clusters with samples very close to the decision boundary of the neighboring clusters. A negative score [-1, 0] indicates that the samples might have got assigned to the wrong clusters." - https://dzone.com/articles/kmeans-silhouette-score-explained-with-python-exam

✅ Knowledge Check - use this moment to stretch students' knowledge with open questions


## 🚀Challenge

Spend some time with this notebook, tweaking parameters. Can you improve the accuracy of the model by cleaning  the data more (removing outliers, for example)? What else can you do to create better clusters?
## [Post-lecture quiz](https://jolly-sea-0a877260f.azurestaticapps.net/quiz/28/)
## Review & Self Study

Take a look at Stanford's K-Means Simulator [here](https://stanford.edu/class/engr108/visualizations/kmeans/kmeans.html). You can use this tool to visualize sample data points and determine its centroids. With fresh data, click 'update' to see how long it takes to find convergence. You can edit the data's randomness, numbers of clusters and numbers of centroids. Does this help you get an idea of how the data can be grouped?

Also, take a look at [this handout on k-means](https://stanford.edu/~cpiech/cs221/handouts/kmeans.html
) from Stanford

## Assignment 

[Try different clustering methods](assignment.md)
