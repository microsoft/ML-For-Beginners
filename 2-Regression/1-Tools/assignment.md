# Regression with Scikit-learn

## Instructions

Take a look at the [Linnerud dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_linnerud.html#sklearn.datasets.load_linnerud) in Scikit-learn. This dataset has multiple [targets](https://scikit-learn.org/stable/datasets/toy_dataset.html#linnerrud-dataset): 'It consists of three exercise (data) and three physiological (target) variables collected from twenty middle-aged men in a fitness club'.

In your own words, describe how to create a Regression model that would plot the relationship between the waistline and how many situps are accomplished. Do the same for the other datapoints in this dataset.

I would load the data in the column at index 1 (situps)  as a numeric predictive value and the column at index 1 (waistline) as predictive target. I would split the sets in 2/3rds for training and 1/3rd for test. I would plot the resulkts of predictions against test values to confirm the corelation between situps and waistline - can the number of sitpus predict the waistline of a person.


## Rubric

| Criteria                       | Exemplary                           | Adequate                      | Needs Improvement          |
| ------------------------------ | ----------------------------------- | ----------------------------- | -------------------------- |
| Submit a descriptive paragraph | Well-written paragraph is submitted | A few sentences are submitted | No description is supplied |
