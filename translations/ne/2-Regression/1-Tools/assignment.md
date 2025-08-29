<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "74a5cf83e4ebc302afbcbc4f418afd0a",
  "translation_date": "2025-08-29T16:51:42+00:00",
  "source_file": "2-Regression/1-Tools/assignment.md",
  "language_code": "ne"
}
-->
# Scikit-learn सँग Regression

## निर्देशनहरू

Scikit-learn मा [Linnerud dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_linnerud.html#sklearn.datasets.load_linnerud) हेर्नुहोस्। यो dataset मा धेरै [targets](https://scikit-learn.org/stable/datasets/toy_dataset.html#linnerrud-dataset) छन्: 'यसमा तीन exercise (data) र तीन physiological (target) variables छन्, जुन बीस जना मध्यम उमेरका पुरुषहरूबाट fitness club मा सङ्कलन गरिएको हो।'

आफ्नै शब्दमा, Regression model कसरी बनाउने भनेर वर्णन गर्नुहोस्, जसले waistline र कति situps गर्न सकिन्छ भन्ने सम्बन्धलाई plot गर्छ। यस dataset का अन्य datapoints का लागि पनि यस्तै गर्नुहोस्।

## मूल्याङ्कन मापदण्ड

| मापदण्ड                       | उत्कृष्ट                           | पर्याप्त                      | सुधार आवश्यक          |
| ------------------------------ | ----------------------------------- | ----------------------------- | -------------------------- |
| वर्णनात्मक अनुच्छेद पेश गर्नुहोस् | राम्रोसँग लेखिएको अनुच्छेद पेश गरिएको छ | केही वाक्यहरू पेश गरिएको छ | कुनै वर्णन पेश गरिएको छैन |

---

**अस्वीकरण**:  
यो दस्तावेज़ AI अनुवाद सेवा [Co-op Translator](https://github.com/Azure/co-op-translator) प्रयोग गरेर अनुवाद गरिएको छ। हामी शुद्धताको लागि प्रयास गर्छौं, तर कृपया ध्यान दिनुहोस् कि स्वचालित अनुवादमा त्रुटिहरू वा अशुद्धताहरू हुन सक्छ। यसको मूल भाषा मा रहेको मूल दस्तावेज़लाई आधिकारिक स्रोत मानिनुपर्छ। महत्वपूर्ण जानकारीको लागि, व्यावसायिक मानव अनुवाद सिफारिस गरिन्छ। यस अनुवादको प्रयोगबाट उत्पन्न हुने कुनै पनि गलतफहमी वा गलत व्याख्याको लागि हामी जिम्मेवार हुने छैनौं।