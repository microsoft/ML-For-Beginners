---
title: 'Build a regression model: prepare and visualize data'
output:
  html_document:
    df_print: paged
    theme: flatly
    highlight: breezedark
    toc: yes
    toc_float: yes
    code_download: yes
---

## **Linear Regression for Pumpkins - Lesson 2**

#### Introduction

Now that you are set up with the tools you need to start tackling machine learning model building with Tidymodels and the Tidyverse, you are ready to start asking questions of your data. As you work with data and apply ML solutions, it's very important to understand how to ask the right question to properly unlock the potentials of your dataset.

In this lesson, you will learn:

-   How to prepare your data for model-building.

-   How to use `ggplot2` for data visualization.

The question you need answered will determine what type of ML algorithms you will leverage. And the quality of the answer you get back will be heavily dependent on the nature of your data.

Let's see this by working through a practical exercise.

![Artwork by \@allison_horst](../../images/unruly_data.jpg){width="700"}

## 1. Importing pumpkins data and summoning the Tidyverse

We'll require the following packages to slice and dice this lesson:

-   `tidyverse`: The [tidyverse](https://www.tidyverse.org/) is a [collection of R packages](https://www.tidyverse.org/packages) designed to makes data science faster, easier and more fun!

You can have them installed as:

`install.packages(c("tidyverse"))`

The script below checks whether you have the packages required to complete this module and installs them for you in case they are missing.

```{r, message=F, warning=F}
if (!require("pacman")) install.packages("pacman")
pacman::p_load(tidyverse)
```

Now, let's fire up some packages and load the [data](https://github.com/microsoft/ML-For-Beginners/blob/main/2-Regression/data/US-pumpkins.csv) provided for this lesson!

```{r load_tidy_verse_models, message=F, warning=F}
# Load the core Tidyverse packages
library(tidyverse)

# Import the pumpkins data
pumpkins <- read_csv(file = "https://raw.githubusercontent.com/microsoft/ML-For-Beginners/main/2-Regression/data/US-pumpkins.csv")


# Get a glimpse and dimensions of the data
glimpse(pumpkins)


# Print the first 50 rows of the data set
pumpkins %>% 
  slice_head(n =50)

```

A quick `glimpse()` immediately shows that there are blanks and a mix of strings (`chr`) and numeric data (`dbl`). The `Date` is of type character and there's also a strange column called `Package` where the data is a mix between `sacks`, `bins` and other values. The data, in fact, is a bit of a mess 😤.

In fact, it is not very common to be gifted a dataset that is completely ready to use to create a ML model out of the box. But worry not, in this lesson, you will learn how to prepare a raw dataset using standard R libraries 🧑‍🔧. You will also learn various techniques to visualize the data.📈📊



> A refresher: The pipe operator (`%>%`) performs operations in logical sequence by passing an object forward into a function or call expression. You can think of the pipe operator as saying "and then" in your code.


## 2. Check for missing data

One of the most common issues data scientists need to deal with is incomplete or missing data. R represents missing, or unknown values, with special sentinel value: `NA` (Not Available).

So how would we know that the data frame contains missing values?

-   One straight forward way would be to use the base R function `anyNA` which returns the logical objects `TRUE` or `FALSE`

```{r anyNA, message=F, warning=F}
pumpkins %>% 
  anyNA()
```

Great, there seems to be some missing data! That's a good place to start.

-   Another way would be to use the function `is.na()` that indicates which individual column elements are missing with a logical `TRUE`.

```{r is_na, message=F, warning=F}
pumpkins %>% 
  is.na() %>% 
  head(n = 7)
```

Okay, got the job done but with a large data frame such as this, it would be inefficient and practically impossible to review all of the rows and columns individually😴.

-   A more intuitive way would be to calculate the sum of the missing values for each column:

```{r colSum_NA, message=F, warning=F}
pumpkins %>% 
  is.na() %>% 
  colSums()
```

Much better! There is missing data, but maybe it won't matter for the task at hand. Let's see what further analysis brings forth.

> Along with the awesome sets of packages and functions, R has a very good documentation. For instance, use `help(colSums)` or `?colSums` to find out more about the function.

## 3. Dplyr: A Grammar of Data Manipulation

![Artwork by \@allison_horst](../../images/dplyr_wrangling.png){width="569"}

[`dplyr`](https://dplyr.tidyverse.org/), a package in the Tidyverse, is a grammar of data manipulation that provides a consistent set of verbs that help you solve the most common data manipulation challenges. In this section, we'll explore some of dplyr's verbs!

#### dplyr::select()

`select()` is a function in the package `dplyr` which helps you pick columns to keep or exclude.

To make your data frame easier to work with, drop several of its columns, using `select()`, keeping only the columns you need.

For instance, in this exercise, our analysis will involve the columns `Package`, `Low Price`, `High Price` and `Date`. Let's select these columns.

```{r select, message=F, warning=F}
# Select desired columns
pumpkins <- pumpkins %>% 
  select(Package, `Low Price`, `High Price`, Date)


# Print data set
pumpkins %>% 
  slice_head(n = 5)
```

#### dplyr::mutate()

`mutate()` is a function in the package `dplyr` which helps you create or modify columns, while keeping the existing columns.

The general structure of mutate is:

`data %>%   mutate(new_column_name = what_it_contains)`

Let's take `mutate` out for a spin using the `Date` column by doing the following operations:

1.  Convert the dates (currently of type character) to a month format (these are US dates, so the format is `MM/DD/YYYY`).

2.  Extract the month from the dates to a new column.

In R, the package [lubridate](https://lubridate.tidyverse.org/) makes it easier to work with Date-time data. So, let's use `dplyr::mutate()`, `lubridate::mdy()`, `lubridate::month()` and see how to achieve the above objectives. We can drop the Date column since we won't be needing it again in subsequent operations.

```{r mut_date, message=F, warning=F}
# Load lubridate
library(lubridate)

pumpkins <- pumpkins %>% 
  # Convert the Date column to a date object
  mutate(Date = mdy(Date)) %>% 
  # Extract month from Date
  mutate(Month = month(Date)) %>% 
  # Drop Date column
  select(-Date)

# View the first few rows
pumpkins %>% 
  slice_head(n = 7)
```

Woohoo! 🤩

Next, let's create a new column `Price`, which represents the average price of a pumpkin. Now, let's take the average of the `Low Price` and `High Price` columns to populate the new Price column.

```{r price, message=F, warning=F}
# Create a new column Price
pumpkins <- pumpkins %>% 
  mutate(Price = (`Low Price` + `High Price`)/2)

# View the first few rows of the data
pumpkins %>% 
  slice_head(n = 5)
```

Yeees!💪

"But wait!", you'll say after skimming through the whole data set with `View(pumpkins)`, "There's something odd here!"🤔

If you look at the `Package` column, pumpkins are sold in many different configurations. Some are sold in `1 1/9 bushel` measures, and some in `1/2 bushel` measures, some per pumpkin, some per pound, and some in big boxes with varying widths.

Let's verify this:

```{r Package, message=F, warning=F}
# Verify the distinct observations in Package column
pumpkins %>% 
  distinct(Package)

```

Amazing!👏

Pumpkins seem to be very hard to weigh consistently, so let's filter them by selecting only pumpkins with the string *bushel* in the `Package` column and put this in a new data frame `new_pumpkins`.

#### dplyr::filter() and stringr::str_detect()

[`dplyr::filter()`](https://dplyr.tidyverse.org/reference/filter.html): creates a subset of the data only containing **rows** that satisfy your conditions, in this case, pumpkins with the string *bushel* in the `Package` column.

[stringr::str_detect()](https://stringr.tidyverse.org/reference/str_detect.html): detects the presence or absence of a pattern in a string.

The [`stringr`](https://github.com/tidyverse/stringr) package provides simple functions for common string operations.

```{r filter, message=F, warning=F}
# Retain only pumpkins with "bushel"
new_pumpkins <- pumpkins %>% 
       filter(str_detect(Package, "bushel"))

# Get the dimensions of the new data
dim(new_pumpkins)

# View a few rows of the new data
new_pumpkins %>% 
  slice_head(n = 5)
```

You can see that we have narrowed down to 415 or so rows of data containing pumpkins by the bushel.🤩

#### dplyr::case_when()

**But wait! There's one more thing to do**

Did you notice that the bushel amount varies per row? You need to normalize the pricing so that you show the pricing per bushel, not per 1 1/9 or 1/2 bushel. Time to do some math to standardize it.

We'll use the function [`case_when()`](https://dplyr.tidyverse.org/reference/case_when.html) to *mutate* the Price column depending on some conditions. `case_when` allows you to vectorise multiple `if_else()`statements.

```{r normalize_price, message=F, warning=F}
# Convert the price if the Package contains fractional bushel values
new_pumpkins <- new_pumpkins %>% 
  mutate(Price = case_when(
    str_detect(Package, "1 1/9") ~ Price/(1 + 1/9),
    str_detect(Package, "1/2") ~ Price/(1/2),
    TRUE ~ Price))

# View the first few rows of the data
new_pumpkins %>% 
  slice_head(n = 30)
```

Now, we can analyze the pricing per unit based on their bushel measurement. All this study of bushels of pumpkins, however, goes to show how very `important` it is to `understand the nature of your data`!

> ✅ According to [The Spruce Eats](https://www.thespruceeats.com/how-much-is-a-bushel-1389308), a bushel's weight depends on the type of produce, as it's a volume measurement. "A bushel of tomatoes, for example, is supposed to weigh 56 pounds... Leaves and greens take up more space with less weight, so a bushel of spinach is only 20 pounds." It's all pretty complicated! Let's not bother with making a bushel-to-pound conversion, and instead price by the bushel. All this study of bushels of pumpkins, however, goes to show how very important it is to understand the nature of your data!
>
> ✅ Did you notice that pumpkins sold by the half-bushel are very expensive? Can you figure out why? Hint: little pumpkins are way pricier than big ones, probably because there are so many more of them per bushel, given the unused space taken by one big hollow pie pumpkin.

Now lastly, for the sheer sake of adventure 💁‍♀️, let's also move the Month column to the first position i.e `before` column `Package`.

`dplyr::relocate()` is used to change column positions.

```{r new_pumpkins, message=F, warning=F}
# Create a new data frame new_pumpkins
new_pumpkins <- new_pumpkins %>% 
  relocate(Month, .before = Package)

new_pumpkins %>% 
  slice_head(n = 7)
  
```

Good job!👌 You now have a clean, tidy dataset on which you can build your new regression model!

## 4. Data visualization with ggplot2

![Infographic by Dasani Madipalli](../../images/data-visualization.png){width="600"}

There is a *wise* saying that goes like this:

> "The simple graph has brought more information to the data analyst's mind than any other device." --- John Tukey

Part of the data scientist's role is to demonstrate the quality and nature of the data they are working with. To do this, they often create interesting visualizations, or plots, graphs, and charts, showing different aspects of data. In this way, they are able to visually show relationships and gaps that are otherwise hard to uncover.

Visualizations can also help determine the machine learning technique most appropriate for the data. A scatterplot that seems to follow a line, for example, indicates that the data is a good candidate for a linear regression exercise.

R offers a number of several systems for making graphs, but [`ggplot2`](https://ggplot2.tidyverse.org/index.html) is one of the most elegant and most versatile. `ggplot2` allows you to compose graphs by **combining independent components**.

Let's start with a simple scatter plot for the Price and Month columns.

So in this case, we'll start with [`ggplot()`](https://ggplot2.tidyverse.org/reference/ggplot.html), supply a dataset and aesthetic mapping (with [`aes()`](https://ggplot2.tidyverse.org/reference/aes.html)) then add a layers (like [`geom_point()`](https://ggplot2.tidyverse.org/reference/geom_point.html)) for scatter plots.

```{r scatter_plt, message=F, warning=F}
# Set a theme for the plots
theme_set(theme_light())

# Create a scatter plot
p <- ggplot(data = new_pumpkins, aes(x = Price, y = Month))
p + geom_point()
```

Is this a useful plot 🤷? Does anything about it surprise you?

It's not particularly useful as all it does is display in your data as a spread of points in a given month.

### **How do we make it useful?**

To get charts to display useful data, you usually need to group the data somehow. For instance in our case, finding the average price of pumpkins for each month would provide more insights to the underlying patterns in our data. This leads us to one more **dplyr** flyby:

#### `dplyr::group_by() %>% summarize()`

Grouped aggregation in R can be easily computed using

`dplyr::group_by() %>% summarize()`

-   `dplyr::group_by()` changes the unit of analysis from the complete dataset to individual groups such as per month.

-   `dplyr::summarize()` creates a new data frame with one column for each grouping variable and one column for each of the summary statistics that you have specified.

For example, we can use the `dplyr::group_by() %>% summarize()` to group the pumpkins into groups based on the **Month** columns and then find the **mean price** for each month.

```{r grp_sumry, message=F, warning=F}
# Find the average price of pumpkins per month
new_pumpkins %>%
  group_by(Month) %>% 
  summarise(mean_price = mean(Price))
```

Succinct!✨

Categorical features such as months are better represented using a bar plot 📊. The layers responsible for bar charts are `geom_bar()` and `geom_col()`. Consult

`?geom_bar` to find out more.

Let's whip up one!

```{r bar_plt, message=F, warning=F}
# Find the average price of pumpkins per month then plot a bar chart
new_pumpkins %>%
  group_by(Month) %>% 
  summarise(mean_price = mean(Price)) %>% 
  ggplot(aes(x = Month, y = mean_price)) +
  geom_col(fill = "midnightblue", alpha = 0.7) +
  ylab("Pumpkin Price")
```

🤩🤩This is a more useful data visualization! It seems to indicate that the highest price for pumpkins occurs in September and October. Does that meet your expectation? Why or why not?

Congratulations on finishing the second lesson 👏! You prepared your data for model building, then uncovered more insights using visualizations!
