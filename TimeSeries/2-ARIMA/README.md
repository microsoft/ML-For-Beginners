# Time Series Forecasting with ARIMA

[![Introduction to ARIMA](https://img.youtube.com/vi/IUSk-YDau10/0.jpg)](https://youtu.be/IUSk-YDau10 "Introduction to ARIMA")

> A brief introduction to ARIMA models. The example is done in R, but the concepts are universal.
## [Pre-lecture quiz](link-to-quiz-app)

In the previous lesson, you learned a bit about Time Series Forecasting and loaded a dataset showing the fluctuations of electrical load over a time period. In this lesson, you will discover a specific way to build models with [ARIMA: *A*uto*R*egressive *I*ntegrated *M*oving *A*verage](https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average). ARIMA models are particularly suited to fit data that shows [non-stationarity](https://en.wikipedia.org/wiki/Stationary_process).

> 🎓 Stationarity, from a statistical context, refers to data whose distribution does not change when shifted in time. Non-stationary data, then, shows fluctuations due to trends that must be transformed to be analyzed. Seasonality, for example, can introduce fluctuations in data and can be eliminated by a process of 'seasonal-differencing'. 

> 🎓 [Differencing](https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average#Differencing) data, again from a statistical context, refers to the process of transforming non-stationary data to make it stationary by removing its non-constant trend. "Differencing removes the changes in the level of a time series, eliminating trend and seasonality and consequently stabilizing the mean of the time series."[Paper by Shixiong et al](https://arxiv.org/abs/1904.07632) 

Let's unpack the parts of ARIMA to better understand how it helps us model Time Series and help us make predictions against it.
## AR - for AutoRegressive

Autoregressive models, as the name implies, look 'back' in time to analyze previous values in your data and make assumptions about them. These previous values are called 'lags'. An example would be data that shows monthly sales of pencils. Each month's sales total would be considered an 'evolving variable' in the dataset. This model is built as the "evolving variable of interest is regressed on its own lagged (i.e., prior) values." [wikipedia](https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average) 
## I - for Integrated

As opposed to the similar 'ARMA' models, the 'I' in ARIMA refers to its *[integrated](https://en.wikipedia.org/wiki/Order_of_integration)* aspect. The data is 'integrated' when differencing steps are applied so as to eliminate non-stationarity.
## MA -  for Moving Average

The [moving-average](https://en.wikipedia.org/wiki/Moving-average_model) aspect of this model refers to the output variable that is determined by observing the current and past values of lags.

Bottom line: ARIMA is used to make a model fit the special form of time series data as closely as possible.
### Preparation

Open the `/working` folder in this lesson and find the `notebook.ipynb` file. We have already loaded 

---

[Step through content in blocks]

## [Topic 1]

### Task:

Work together to progressively enhance your codebase to build the project with shared code:

```html
code blocks
```

✅ Knowledge Check - use this moment to stretch students' knowledge with open questions

## [Topic 2]

## [Topic 3]

## 🚀Challenge

Add a challenge for students to work on collaboratively in class to enhance the project

Optional: add a screenshot of the completed lesson's UI if appropriate

## [Post-lecture quiz](link-to-quiz-app)

## Review & Self Study

**Assignment**: [Assignment Name](assignment.md)
