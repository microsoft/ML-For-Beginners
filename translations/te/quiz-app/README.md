<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "6d130dffca5db70d7e615f926cb1ad4c",
  "translation_date": "2025-12-19T13:00:27+00:00",
  "source_file": "quiz-app/README.md",
  "language_code": "te"
}
-->
# క్విజ్‌లు

ఈ క్విజ్‌లు https://aka.ms/ml-beginners వద్ద ML పాఠ్యక్రమం కోసం ప్రీ- మరియు పోస్ట్-లెక్చర్ క్విజ్‌లు.

## ప్రాజెక్ట్ సెటప్

```
npm install
```

### అభివృద్ధి కోసం కంపైల్ చేసి హాట్-రిలోడ్ చేస్తుంది

```
npm run serve
```

### ఉత్పత్తి కోసం కంపైల్ చేసి మినిఫై చేస్తుంది

```
npm run build
```

### ఫైళ్లను లింట్ చేసి సరిచేస్తుంది

```
npm run lint
```

### కాన్ఫిగరేషన్‌ను అనుకూలీకరించండి

[Configuration Reference](https://cli.vuejs.org/config/) చూడండి.

క్రెడిట్స్: ఈ క్విజ్ యాప్ యొక్క అసలు వెర్షన్‌కు ధన్యవాదాలు: https://github.com/arpan45/simple-quiz-vue

## Azureకి డిప్లాయ్ చేయడం

మీరు ప్రారంభించడానికి సహాయపడే దశల వారీ గైడ్ ఇక్కడ ఉంది:

1. GitHub రిపాజిటరీని ఫోర్క్ చేయండి  
మీ స్టాటిక్ వెబ్ యాప్ కోడ్ మీ GitHub రిపాజిటరీలో ఉందని నిర్ధారించుకోండి. ఈ రిపాజిటరీని ఫోర్క్ చేయండి.

2. Azure స్టాటిక్ వెబ్ యాప్ సృష్టించండి  
- [Azure ఖాతా](http://azure.microsoft.com) సృష్టించండి  
- [Azure పోర్టల్](https://portal.azure.com) కు వెళ్లండి  
- "Create a resource" పై క్లిక్ చేసి "Static Web App" కోసం శోధించండి.  
- "Create" పై క్లిక్ చేయండి.

3. స్టాటిక్ వెబ్ యాప్‌ను కాన్ఫిగర్ చేయండి  
- ప్రాథమికాలు: సబ్‌స్క్రిప్షన్: మీ Azure సబ్‌స్క్రిప్షన్‌ను ఎంచుకోండి.  
- రిసోర్స్ గ్రూప్: కొత్త రిసోర్స్ గ్రూప్ సృష్టించండి లేదా ఉన్నదాన్ని ఉపయోగించండి.  
- పేరు: మీ స్టాటిక్ వెబ్ యాప్‌కు పేరు ఇవ్వండి.  
- ప్రాంతం: మీ వినియోగదారులకు సమీప ప్రాంతాన్ని ఎంచుకోండి.

- #### డిప్లాయ్‌మెంట్ వివరాలు:  
- మూలం: "GitHub" ఎంచుకోండి.  
- GitHub ఖాతా: Azureకి మీ GitHub ఖాతాకు యాక్సెస్ అనుమతించండి.  
- సంస్థ: మీ GitHub సంస్థను ఎంచుకోండి.  
- రిపాజిటరీ: మీ స్టాటిక్ వెబ్ యాప్ ఉన్న రిపాజిటరీని ఎంచుకోండి.  
- బ్రాంచ్: మీరు డిప్లాయ్ చేయదలచుకున్న బ్రాంచ్‌ను ఎంచుకోండి.

- #### బిల్డ్ వివరాలు:  
- బిల్డ్ ప్రీసెట్‌లు: మీ యాప్ నిర్మించబడిన ఫ్రేమ్‌వర్క్‌ను ఎంచుకోండి (ఉదా: React, Angular, Vue, మొదలైనవి).  
- యాప్ లొకేషన్: మీ యాప్ కోడ్ ఉన్న ఫోల్డర్‌ను పేర్కొనండి (ఉదా: / రూట్‌లో ఉంటే).  
- API లొకేషన్: మీకు API ఉంటే, దాని స్థానం (ఐచ్ఛికం) పేర్కొనండి.  
- అవుట్‌పుట్ లొకేషన్: బిల్డ్ అవుట్‌పుట్ ఉత్పత్తి అయ్యే ఫోల్డర్‌ను పేర్కొనండి (ఉదా: build లేదా dist).

4. సమీక్షించి సృష్టించండి  
మీ సెట్టింగ్స్‌ను సమీక్షించి "Create" పై క్లిక్ చేయండి. Azure అవసరమైన వనరులను సెట్ చేసి మీ రిపాజిటరీలో GitHub Actions వర్క్‌ఫ్లోని సృష్టిస్తుంది.

5. GitHub Actions వర్క్‌ఫ్లో  
Azure మీ రిపాజిటరీలో (.github/workflows/azure-static-web-apps-<name>.yml) GitHub Actions వర్క్‌ఫ్లో ఫైల్‌ను ఆటోమేటిక్‌గా సృష్టిస్తుంది. ఈ వర్క్‌ఫ్లో బిల్డ్ మరియు డిప్లాయ్‌మెంట్ ప్రక్రియను నిర్వహిస్తుంది.

6. డిప్లాయ్‌మెంట్‌ను మానిటర్ చేయండి  
మీ GitHub రిపాజిటరీలో "Actions" ట్యాబ్‌కు వెళ్లండి.  
ఒక వర్క్‌ఫ్లో నడుస్తున్నట్లు మీరు చూడగలరు. ఈ వర్క్‌ఫ్లో మీ స్టాటిక్ వెబ్ యాప్‌ను Azureకి బిల్డ్ చేసి డిప్లాయ్ చేస్తుంది.  
వర్క్‌ఫ్లో పూర్తయిన తర్వాత, మీ యాప్ అందించిన Azure URLపై లైవ్ అవుతుంది.

### ఉదాహరణ వర్క్‌ఫ్లో ఫైల్

GitHub Actions వర్క్‌ఫ్లో ఫైల్ ఎలా ఉండొచ్చో ఒక ఉదాహరణ ఇక్కడ ఉంది:  
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
  
### అదనపు వనరులు  
- [Azure Static Web Apps డాక్యుమెంటేషన్](https://learn.microsoft.com/azure/static-web-apps/getting-started)  
- [GitHub Actions డాక్యుమెంటేషన్](https://docs.github.com/actions/use-cases-and-examples/deploying/deploying-to-azure-static-web-app)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**అస్పష్టత**:  
ఈ పత్రాన్ని AI అనువాద సేవ [Co-op Translator](https://github.com/Azure/co-op-translator) ఉపయోగించి అనువదించబడింది. మేము ఖచ్చితత్వానికి ప్రయత్నించినప్పటికీ, ఆటోమేటెడ్ అనువాదాల్లో పొరపాట్లు లేదా తప్పిదాలు ఉండవచ్చు. అసలు పత్రం దాని స్వదేశీ భాషలోనే అధికారిక మూలంగా పరిగణించాలి. ముఖ్యమైన సమాచారానికి, ప్రొఫెషనల్ మానవ అనువాదం సిఫార్సు చేయబడుతుంది. ఈ అనువాదం వలన కలిగే ఏవైనా అపార్థాలు లేదా తప్పుదారుల బాధ్యత మేము తీసుకోము.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->