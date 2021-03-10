# Build a Regression Model using Scikit-Learn: Prepare and Visualize Data

> Sketchnote
## [Pre-lecture quiz](https://jolly-sea-0a877260f.azurestaticapps.net/quiz/7/)

### Introduction

In this lesson, you will learn:
- Preparing your data for model-building
- Two data visualization techniques and libraries
### Asking the Right Question

As you work with data and apply ML solutions, it's very important to understand how to ask the right question to properly unlock the potentials of your dataset. 

The question you need answered will determine what type of ML algorithms you will leverage. For example, do you need to determine the differences between cars and trucks as they cruise down a highway via a video feed? You will need some kind of highly performant classification model to make that differentiation. It will need to be able to perform object detection, probably by showing bounding boxes around detected cars and trucks.

> infographic here

What if you are trying to correlate two points of data - like age to height? You can use a regression model, as shown in the previous lesson, to draw the classical straight line through the scatterplot of points to show how, with age, height tends to increase. Thus you can predict, for a given group of people, their height given their age.

> infographic here

But it's not very common to be gifted a dataset that is completely ready to use to create a ML model. In this lesson, you will learn how to prepare a raw dataset using standard Python libraries. You will also learn various techniques to visualize the data.
### Preparation

In this folder you will find a .csv file called `US-pumpkins.csv` which includes 1757 lines of data about the pumpkin market, sorted into groupings by city. This is the raw data extracted from the [Specialty Crops Terminal Markets Standard Reports](https://www.marketnews.usda.gov/mnp/fv-report-config-step1?type=termPrice) distributed by the United States Department of Agriculture. 

This data is in the public domain. It can be downloaded in many separate files, per city, from the USDA web site. To avoid too many separate files we have concatenated all the city data into one spreadsheet. Take a look at this file.
## The Pumpkin data

What do you notice about this data? First, you see that it is a mix of text and numeric data. There are also dates. Second, you see that there's a considerable amount of missing and mixed data. To build a good model, you will need to handle that. 

What question can you ask of this data, using a Regression technique? What about "Predict the price of a pumpkin for sale during a given month". Looking again at the data, there are some changes you need to make to create the data structure necessary for the task. 
### Analyze the Pumpkin Data

Let's use [Pandas](https://pandas.pydata.org/), (the name stands for `Python Data Analysis`) a tool very useful for shaping data, to analyze and prepare this pumpkin data. First, check for missing dates and then convert the dates to a month format (these are US dates, so the format is currently `MM/DD/YYYY`). Extract the month to a new column.

Open the `notebook.ipynb` file in VS Code and import the spreadsheet in to a new Pandas dataframe. Use the `head()` function to view the first five rows.

```python
import pandas as pd
pumpkins = pd.read_csv('US-pumpkins.csv')
pumpkins.head()
```

✅ What function would you use to view the last five rows?

Check if there is missing data in the current dataframe:

```python
pumpkins.isnull().sum()
```

There is missing data, but maybe it won't matter for the task at hand.

To make your dataframe easier to work with, drop several of its columns, keeping only the ones you need: 

```python
new_columns = ['Package', 'Month', 'Low Price', 'High Price', 'Date']
pumpkins = pumpkins.drop([c for c in pumpkins.columns if c not in new_columns], axis=1)
```

Second, think about how to determine the average price of a pumpkin in a given month. What columns would you pick for this task? Hint: you'll need 3 columns.

Solution: take the average of the Low Price and High Price columns to populate the new Price column, and convert the Date column to only show the month. Fortunately, according to the check above, there is no missing data for dates or prices.

```python
price = (pumpkins['Low Price'] + pumpkins['High Price']) / 2

month = pd.DatetimeIndex(pumpkins['Date']).month

```
✅ Feel free to print any data you'd like to check: `print(month)` for example.

Now, append your converted data into a fresh Pandas dataframe:

```python
new_pumpkins = pd.DataFrame({'Month': month, 'Package': pumpkins['Package'], 'Low Price': pumpkins['Low Price'],'High Price': pumpkins['High Price'], 'Price': price})
```
Printing out your dataframe will show you a clean, tidy dataset on which you can build your new regression model.

But wait! There's something odd here. If you look at the `Package` column, pumpkins are sold in many different configurations. Some are sold in '1 1/9 bushel' measures, and some in '1/2 bushel' measures, some per pumpkin, some per pound, and some in big boxes with varying widths.

Digging into the original data, it's interesting that anything with `Unit of Sale` equalling 'EACH' or 'PER BIN' also have the `Package` type per inch, per bin, or 'each'. Pumpkins seem to be very hard to weigh consistently, so let's filter them out by selecting only pumpkins with the string 'bushel' in their `Package` column. Add a filter at the top of the file, under the initial .csv import:

```python
pumpkins = pumpkins[pumpkins['Package'].str.contains('bushel', case=True, regex=True)]
```
If you print the data now, you can see that you are only getting the 415 or so rows of data containing pumpkins by the bushel. But wait! there's one more thing to do. Did you notice that the bushel amount varies per row? You need to normalize the pricing so that you show the pricing per bushel, so do some math to standardize it. Add these lines after the block creating the new_pumpkins dataframe:

```python
new_pumpkins.loc[new_pumpkins['Package'].str.contains('1 1/9'), 'Price'] = price/1.1

new_pumpkins.loc[new_pumpkins['Package'].str.contains('1/2'), 'Price'] = price*2
```

✅ According to [The Spruce Eats](https://www.thespruceeats.com/how-much-is-a-bushel-1389308), a bushel's weight depends on the type of produce, as it's a volume measurement. "A bushel of tomatoes, for example, is supposed to weigh 56 pounds... Leaves and greens take up more space with less weight, so a bushel of spinach is only 20 pounds." It's all pretty complicated! Let's not bother with making a bushel-to-pound conversion, and instead price by the bushel.

Now, you can analyze the pricing per unit based on their bushel measurement. If you print out the data one more time, you can see how it's standardized.

✅ Did you notice that pumpkins sold by the half-bushel are very expensive? Can you figure out why? Hint: little pumpkins are way pricier than big ones, probably because there are so many more of them per bushel, given the unused space taken by one big hollow pie pumpkin.
## Visualization Strategies

Part of the data scientist's role is to demonstrate the quality and nature of the data they are working with. To do this, they often create interesting visualizations, or plots, graphs, and charts, showing different aspects of data. In this way, they are able to visually show relationships and gaps that are otherwise hard to uncover.

One data visualization libary that works well in Jupyter notebooks is [Matplotlib](https://matplotlib.org/) (which you also saw in the previous lesson).

> Get more experience with data visualization in [these tutorials](https://docs.microsoft.com/learn/modules/explore-analyze-data-with-python?WT.mc_id=academic-15963-cxa).
## Experiment with Matplotlib

Try to create some simple plots to display the new dataframe you just created. What would a basic line plot show?

Import Matplotlib at the top of the file, under the Pandas import:

```python
import matplotlib.pyplot as plt
```

Rerun the entire notebook to refresh. Then at the bottom of the notebook, add a cell to plot the data as a box:

```python
new_pumpkins.plot(kind='bar', y='Price')
plt.show()
```
Is this a useful plot? Does anything about it surprise you?

It's not particularly useful, as there are too many numbers in the x axis. All it does is show all the prices in your data. To get charts to display useful data, you usually need to group the data somehow. Let's try creating a plot where the y axis shows the months and the data demonstrates the distribution of data. 

Add a cell to create a grouped bar chart:

```python
new_pumpkins.groupby(['Month'])['Price'].mean().plot(kind='bar')
plt.ylabel("Pumpkin Price")
```
This is a more useful data visualization! It seems to indicate that the highest price for pumpkins occurs in September and October. Does that meet your expectation? Why or why not?

🚀 Challenge: Add a challenge for students to work on collaboratively in class to enhance the project

Optional: add a screenshot of the completed lesson's UI if appropriate

## [Pre-lecture quiz](https://jolly-sea-0a877260f.azurestaticapps.net/quiz/8/)

## Review & Self Study

**Assignment**: [Assignment Name](assignment.md)
