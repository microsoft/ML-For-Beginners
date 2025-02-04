# Quizzes

ये क्विज़ https://aka.ms/ml-beginners पर ML पाठ्यक्रम के लिए प्री- और पोस्ट-लेक्चर क्विज़ हैं।

## प्रोजेक्ट सेटअप

```
npm install
```

### विकास के लिए संकलन और हॉट-रीलोड

```
npm run serve
```

### उत्पादन के लिए संकलन और संक्षिप्त

```
npm run build
```

### फाइलों को लिंट और ठीक करता है

```
npm run lint
```

### कॉन्फ़िगरेशन को कस्टमाइज़ करें

[Configuration Reference](https://cli.vuejs.org/config/) देखें।

श्रेय: इस क्विज़ ऐप के मूल संस्करण के लिए धन्यवाद: https://github.com/arpan45/simple-quiz-vue

## Azure पर तैनाती

यहां एक चरण-दर-चरण मार्गदर्शिका है जो आपको आरंभ करने में मदद करेगी:

1. एक GitHub रिपॉजिटरी को फोर्क करें
सुनिश्चित करें कि आपका स्थिर वेब ऐप कोड आपके GitHub रिपॉजिटरी में है। इस रिपॉजिटरी को फोर्क करें।

2. एक Azure Static Web App बनाएं
- एक [Azure खाता](http://azure.microsoft.com) बनाएं
- [Azure पोर्टल](https://portal.azure.com) पर जाएं 
- “Create a resource” पर क्लिक करें और “Static Web App” खोजें।
- “Create” पर क्लिक करें।

3. Static Web App को कॉन्फ़िगर करें
- मूलभूत जानकारी: Subscription: अपनी Azure सब्सक्रिप्शन चुनें।
- Resource Group: एक नया संसाधन समूह बनाएं या मौजूदा का उपयोग करें।
- नाम: अपने स्थिर वेब ऐप के लिए एक नाम प्रदान करें।
- क्षेत्र: अपने उपयोगकर्ताओं के निकटतम क्षेत्र चुनें।

- #### तैनाती विवरण:
- स्रोत: “GitHub” चुनें।
- GitHub खाता: Azure को आपके GitHub खाते तक पहुंचने के लिए अधिकृत करें।
- संगठन: अपना GitHub संगठन चुनें।
- रिपॉजिटरी: वह रिपॉजिटरी चुनें जिसमें आपका स्थिर वेब ऐप है।
- शाखा: वह शाखा चुनें जिससे आप तैनात करना चाहते हैं।

- #### निर्माण विवरण:
- निर्माण प्रीसेट: उस फ्रेमवर्क को चुनें जिससे आपका ऐप बनाया गया है (उदा., React, Angular, Vue, आदि)।
- ऐप स्थान: उस फ़ोल्डर को निर्दिष्ट करें जिसमें आपका ऐप कोड है (उदा., / यदि यह रूट में है)।
- API स्थान: यदि आपके पास API है, तो उसका स्थान निर्दिष्ट करें (वैकल्पिक)।
- आउटपुट स्थान: उस फ़ोल्डर को निर्दिष्ट करें जहां निर्माण आउटपुट उत्पन्न होता है (उदा., build या dist)।

4. समीक्षा और निर्माण
अपनी सेटिंग्स की समीक्षा करें और “Create” पर क्लिक करें। Azure आवश्यक संसाधनों को सेटअप करेगा और आपके रिपॉजिटरी में एक GitHub Actions वर्कफ़्लो बनाएगा।

5. GitHub Actions वर्कफ़्लो
Azure स्वचालित रूप से आपके रिपॉजिटरी में एक GitHub Actions वर्कफ़्लो फ़ाइल बनाएगा (.github/workflows/azure-static-web-apps-<name>.yml)। यह वर्कफ़्लो निर्माण और तैनाती प्रक्रिया को संभालेगा।

6. तैनाती की निगरानी करें
अपने GitHub रिपॉजिटरी में “Actions” टैब पर जाएं।
आपको एक वर्कफ़्लो चलता हुआ दिखाई देना चाहिए। यह वर्कफ़्लो आपके स्थिर वेब ऐप को Azure पर निर्माण और तैनात करेगा।
एक बार वर्कफ़्लो पूरा हो जाने पर, आपका ऐप प्रदान किए गए Azure URL पर लाइव हो जाएगा।

### उदाहरण वर्कफ़्लो फ़ाइल

यहां एक उदाहरण है कि GitHub Actions वर्कफ़्लो फ़ाइल कैसी दिख सकती है:
name: Azure Static Web Apps CI/CD
```
on:
  push:
    branches:
      - main
  pull_request:
    types: [opened, synchronize, reopened, closed]
    branches:
      - main

jobs:
  build_and_deploy_job:
    runs-on: ubuntu-latest
    name: Build and Deploy Job
    steps:
      - uses: actions/checkout@v2
      - name: Build And Deploy
        id: builddeploy
        uses: Azure/static-web-apps-deploy@v1
        with:
          azure_static_web_apps_api_token: ${{ secrets.AZURE_STATIC_WEB_APPS_API_TOKEN }}
          repo_token: ${{ secrets.GITHUB_TOKEN }}
          action: "upload"
          app_location: "/quiz-app" # App source code path
          api_location: ""API source code path optional
          output_location: "dist" #Built app content directory - optional
```

### अतिरिक्त संसाधन
- [Azure Static Web Apps Documentation](https://learn.microsoft.com/azure/static-web-apps/getting-started)
- [GitHub Actions Documentation](https://docs.github.com/actions/use-cases-and-examples/deploying/deploying-to-azure-static-web-app)

**अस्वीकरण**:
यह दस्तावेज़ मशीन-आधारित एआई अनुवाद सेवाओं का उपयोग करके अनुवादित किया गया है। जबकि हम सटीकता के लिए प्रयास करते हैं, कृपया ध्यान दें कि स्वचालित अनुवाद में त्रुटियाँ या गलतियाँ हो सकती हैं। अपनी मूल भाषा में मूल दस्तावेज़ को प्राधिकृत स्रोत माना जाना चाहिए। महत्वपूर्ण जानकारी के लिए, पेशेवर मानव अनुवाद की सिफारिश की जाती है। इस अनुवाद के उपयोग से उत्पन्न किसी भी गलतफहमी या गलत व्याख्या के लिए हम उत्तरदायी नहीं हैं।