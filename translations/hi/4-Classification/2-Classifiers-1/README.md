# व्यंजन वर्गीकरणकर्ता 1

इस पाठ में, आप पिछले पाठ से सहेजे गए संतुलित और साफ डेटा से भरे डेटासेट का उपयोग करेंगे, जो सभी व्यंजनों के बारे में है।

आप इस डेटासेट का उपयोग विभिन्न वर्गीकरणकर्ताओं के साथ करेंगे ताकि _सामग्री के एक समूह के आधार पर किसी दिए गए राष्ट्रीय व्यंजन की भविष्यवाणी की जा सके_। ऐसा करते समय, आप उन तरीकों के बारे में अधिक जानेंगे जिनसे एल्गोरिदम को वर्गीकरण कार्यों के लिए उपयोग किया जा सकता है।

## [पाठ पूर्व क्विज़](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/21/)
# तैयारी

मान लेते हैं कि आपने [पाठ 1](../1-Introduction/README.md) पूरा कर लिया है, सुनिश्चित करें कि इन चार पाठों के लिए रूट `/data` फ़ोल्डर में एक _cleaned_cuisines.csv_ फ़ाइल मौजूद है।

## अभ्यास - एक राष्ट्रीय व्यंजन की भविष्यवाणी करें

1. इस पाठ के _notebook.ipynb_ फ़ोल्डर में काम करते हुए, उस फ़ाइल को पांडस लाइब्रेरी के साथ आयात करें:

    ```python
    import pandas as pd
    cuisines_df = pd.read_csv("../data/cleaned_cuisines.csv")
    cuisines_df.head()
    ```

    डेटा इस प्रकार दिखता है:

|     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
| --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
| 0   | 0          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 1   | 1          | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 2   | 2          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 3   | 3          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 4   | 4          | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
  

1. अब, कई और लाइब्रेरी आयात करें:

    ```python
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
    from sklearn.svm import SVC
    import numpy as np
    ```

1. प्रशिक्षण के लिए X और y निर्देशांक को दो डेटा फ्रेम में विभाजित करें। `cuisine` लेबल डेटा फ्रेम हो सकता है:

    ```python
    cuisines_label_df = cuisines_df['cuisine']
    cuisines_label_df.head()
    ```

    यह इस प्रकार दिखेगा:

    ```output
    0    indian
    1    indian
    2    indian
    3    indian
    4    indian
    Name: cuisine, dtype: object
    ```

1. उस `Unnamed: 0` column and the `cuisine` column, calling `drop()` को हटा दें। शेष डेटा को ट्रेन करने योग्य फीचर्स के रूप में सहेजें:

    ```python
    cuisines_feature_df = cuisines_df.drop(['Unnamed: 0', 'cuisine'], axis=1)
    cuisines_feature_df.head()
    ```

    आपके फीचर्स इस प्रकार दिखते हैं:

|      | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | artemisia | artichoke |  ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood |  yam | yeast | yogurt | zucchini |
| ---: | -----: | -------: | ----: | ---------: | ----: | -----------: | ------: | -------: | --------: | --------: | ---: | ------: | ----------: | ---------: | ----------------------: | ---: | ---: | ---: | ----: | -----: | -------: |
|    0 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    1 |      1 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |       0 |           0 |          0 |                       0 |    0 |    0 |    0 |     0 |      0 |        0 | 0 |
|    2 |      0 |        0 |     0 |          0 |     0 |            0 |       0 |        0 |         0 |         0 |  ... |  

**अस्वीकरण**:
इस दस्तावेज़ का अनुवाद मशीन-आधारित AI अनुवाद सेवाओं का उपयोग करके किया गया है। जबकि हम सटीकता के लिए प्रयास करते हैं, कृपया ध्यान दें कि स्वचालित अनुवाद में त्रुटियाँ या गलतियाँ हो सकती हैं। अपनी मूल भाषा में मूल दस्तावेज़ को प्राधिकृत स्रोत माना जाना चाहिए। महत्वपूर्ण जानकारी के लिए, पेशेवर मानव अनुवाद की सिफारिश की जाती है। इस अनुवाद के उपयोग से उत्पन्न किसी भी गलतफहमी या गलत व्याख्या के लिए हम उत्तरदायी नहीं हैं।