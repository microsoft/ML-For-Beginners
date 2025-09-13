<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "6d130dffca5db70d7e615f926cb1ad4c",
  "translation_date": "2025-08-29T17:47:45+00:00",
  "source_file": "quiz-app/README.md",
  "language_code": "mr"
}
-->
# क्विझेस

ही क्विझेस ML अभ्यासक्रमासाठीच्या व्याख्यानांपूर्वी आणि नंतर घेण्यात येणाऱ्या क्विझेस आहेत. अधिक माहितीसाठी भेट द्या: https://aka.ms/ml-beginners

## प्रकल्प सेटअप

```
npm install
```

### विकासासाठी संकलन आणि हॉट-रिलोड

```
npm run serve
```

### उत्पादनासाठी संकलन आणि मिनिफिकेशन

```
npm run build
```

### फायली तपासणे आणि दुरुस्त करणे

```
npm run lint
```

### कॉन्फिगरेशन सानुकूलित करा

[Configuration Reference](https://cli.vuejs.org/config/) येथे पहा.

क्रेडिट्स: या क्विझ अॅपच्या मूळ आवृत्तीबद्दल धन्यवाद: https://github.com/arpan45/simple-quiz-vue

## Azure वर डिप्लॉय करणे

येथे सुरुवात करण्यासाठी चरण-दर-चरण मार्गदर्शक दिला आहे:

1. GitHub रेपॉझिटरी फोर्क करा  
तुमच्या GitHub रेपॉझिटरीमध्ये तुमच्या स्थिर वेब अॅपचा कोड असावा. ही रेपॉझिटरी फोर्क करा.

2. Azure Static Web App तयार करा  
- [Azure खाते](http://azure.microsoft.com) तयार करा  
- [Azure पोर्टल](https://portal.azure.com) वर जा  
- “Create a resource” वर क्लिक करा आणि “Static Web App” शोधा.  
- “Create” वर क्लिक करा.  

3. Static Web App कॉन्फिगर करा  
- #### बेसिक्स:  
  - Subscription: तुमची Azure सबस्क्रिप्शन निवडा.  
  - Resource Group: नवीन रिसोर्स ग्रुप तयार करा किंवा विद्यमान वापरा.  
  - Name: तुमच्या स्थिर वेब अॅपसाठी नाव द्या.  
  - Region: तुमच्या वापरकर्त्यांच्या जवळचा प्रदेश निवडा.  

- #### डिप्लॉयमेंट तपशील:  
  - Source: “GitHub” निवडा.  
  - GitHub Account: Azure ला तुमच्या GitHub खात्याचा प्रवेश अधिकृत करा.  
  - Organization: तुमची GitHub संस्था निवडा.  
  - Repository: तुमच्या स्थिर वेब अॅप असलेल्या रेपॉझिटरीची निवड करा.  
  - Branch: ज्या शाखेतून तुम्हाला डिप्लॉय करायचे आहे ती निवडा.  

- #### बिल्ड तपशील:  
  - Build Presets: तुमचे अॅप ज्या फ्रेमवर्कवर आधारित आहे ते निवडा (उदा., React, Angular, Vue, इ.).  
  - App Location: तुमच्या अॅप कोड असलेल्या फोल्डरचे स्थान निर्दिष्ट करा (उदा., / जर ते रूटमध्ये असेल).  
  - API Location: जर तुमच्याकडे API असेल, तर त्याचे स्थान निर्दिष्ट करा (पर्यायी).  
  - Output Location: बिल्ड आउटपुट जिथे तयार होते तो फोल्डर निर्दिष्ट करा (उदा., build किंवा dist).  

4. पुनरावलोकन आणि तयार करा  
तुमच्या सेटिंग्ज पुनरावलोकन करा आणि “Create” वर क्लिक करा. Azure आवश्यक संसाधने सेटअप करेल आणि तुमच्या रेपॉझिटरीमध्ये GitHub Actions वर्कफ्लो तयार करेल.  

5. GitHub Actions वर्कफ्लो  
Azure तुमच्या रेपॉझिटरीमध्ये स्वयंचलितपणे GitHub Actions वर्कफ्लो फाइल तयार करेल (.github/workflows/azure-static-web-apps-<name>.yml). हा वर्कफ्लो बिल्ड आणि डिप्लॉयमेंट प्रक्रिया हाताळेल.  

6. डिप्लॉयमेंट मॉनिटर करा  
तुमच्या GitHub रेपॉझिटरीच्या “Actions” टॅबवर जा.  
तुम्हाला वर्कफ्लो चालू असल्याचे दिसेल. हा वर्कफ्लो तुमचा स्थिर वेब अॅप Azure वर बिल्ड आणि डिप्लॉय करेल.  
वर्कफ्लो पूर्ण झाल्यावर, तुमचे अॅप दिलेल्या Azure URL वर लाइव्ह असेल.  

### वर्कफ्लो फाइलचे उदाहरण

GitHub Actions वर्कफ्लो फाइल कशी दिसू शकते याचे उदाहरण येथे आहे:  
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

### अतिरिक्त संसाधने  
- [Azure Static Web Apps Documentation](https://learn.microsoft.com/azure/static-web-apps/getting-started)  
- [GitHub Actions Documentation](https://docs.github.com/actions/use-cases-and-examples/deploying/deploying-to-azure-static-web-app)  

---

**अस्वीकरण**:  
हा दस्तऐवज AI भाषांतर सेवा [Co-op Translator](https://github.com/Azure/co-op-translator) वापरून भाषांतरित करण्यात आला आहे. आम्ही अचूकतेसाठी प्रयत्नशील असलो तरी कृपया लक्षात ठेवा की स्वयंचलित भाषांतरांमध्ये त्रुटी किंवा अचूकतेचा अभाव असू शकतो. मूळ भाषेतील दस्तऐवज हा अधिकृत स्रोत मानला जावा. महत्त्वाच्या माहितीसाठी व्यावसायिक मानवी भाषांतराची शिफारस केली जाते. या भाषांतराचा वापर करून निर्माण होणाऱ्या कोणत्याही गैरसमज किंवा चुकीच्या अर्थासाठी आम्ही जबाबदार राहणार नाही.