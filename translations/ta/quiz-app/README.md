<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "6d130dffca5db70d7e615f926cb1ad4c",
  "translation_date": "2025-10-11T11:14:10+00:00",
  "source_file": "quiz-app/README.md",
  "language_code": "ta"
}
-->
# வினாடி வினா

இந்த வினாடி வினாக்கள் https://aka.ms/ml-beginners என்ற ML பாடத்திட்டத்திற்கான முன் மற்றும் பிந்தைய பாடவகுப்பு வினாடி வினாக்கள் ஆகும்.

## திட்ட அமைப்பு

```
npm install
```

### மேம்பாட்டிற்காக தொகுத்து, தானாக மீண்டும் ஏற்றுகிறது

```
npm run serve
```

### உற்பத்திக்காக தொகுத்து, சுருக்குகிறது

```
npm run build
```

### கோப்புகளை சரிசெய்து, பிழைகளை திருத்துகிறது

```
npm run lint
```

### அமைப்பை தனிப்பயனாக்கவும்

[Configuration Reference](https://cli.vuejs.org/config/) ஐப் பார்க்கவும்.

கிரெடிட்ஸ்: இந்த வினாடி வினா செயலியின் அசல் பதிப்புக்கு நன்றி: https://github.com/arpan45/simple-quiz-vue

## Azure-க்கு வெளியிடுதல்

இங்கே நீங்கள் தொடங்க உதவும் படிப்படியான வழிகாட்டி உள்ளது:

1. GitHub களஞ்சியத்தை Fork செய்யவும்  
உங்கள் நிலையான வலை செயலி குறியீடு உங்கள் GitHub களஞ்சியத்தில் இருக்க வேண்டும். இந்த களஞ்சியத்தை Fork செய்யவும்.

2. Azure Static Web App ஒன்றை உருவாக்கவும்  
- [Azure கணக்கை](http://azure.microsoft.com) உருவாக்கவும்  
- [Azure portal](https://portal.azure.com) க்கு செல்லவும்  
- “Create a resource” என்பதைக் கிளிக் செய்து “Static Web App” ஐத் தேடவும்.  
- “Create” என்பதைக் கிளிக் செய்யவும்.  

3. Static Web App ஐ அமைக்கவும்  
- அடிப்படை: Subscription: உங்கள் Azure சந்தாவைத் தேர்ந்தெடுக்கவும்.  
- Resource Group: புதிய resource group ஒன்றை உருவாக்கவும் அல்லது ஏற்கனவே உள்ள ஒன்றைப் பயன்படுத்தவும்.  
- Name: உங்கள் static web app க்கு ஒரு பெயரை வழங்கவும்.  
- Region: உங்கள் பயனர்களுக்கு அருகிலுள்ள பகுதியைத் தேர்ந்தெடுக்கவும்.  

- #### Deployment விவரங்கள்:  
- Source: “GitHub” ஐத் தேர்ந்தெடுக்கவும்.  
- GitHub Account: Azure உங்கள் GitHub கணக்கத்தை அணுக அனுமதிக்கவும்.  
- Organization: உங்கள் GitHub அமைப்பைத் தேர்ந்தெடுக்கவும்.  
- Repository: உங்கள் static web app உள்ள களஞ்சியத்தைத் தேர்ந்தெடுக்கவும்.  
- Branch: நீங்கள் வெளியிட விரும்பும் கிளையைத் தேர்ந்தெடுக்கவும்.  

- #### Build விவரங்கள்:  
- Build Presets: உங்கள் செயலி உருவாக்கப்பட்டுள்ள கட்டமைப்பைத் தேர்ந்தெடுக்கவும் (எ.கா., React, Angular, Vue, முதலியன).  
- App Location: உங்கள் செயலி குறியீடு உள்ள கோப்புறையை குறிப்பிடவும் (எ.கா., / இது root இல் இருந்தால்).  
- API Location: உங்கள் செயலியில் API இருந்தால், அதன் இருப்பிடத்தை குறிப்பிடவும் (விருப்பத்தேர்வு).  
- Output Location: உருவாக்கப்பட்ட வெளியீட்டு கோப்புகள் உருவாகும் கோப்புறையை குறிப்பிடவும் (எ.கா., build அல்லது dist).  

4. மதிப்பாய்வு செய்து உருவாக்கவும்  
உங்கள் அமைப்புகளை மதிப்பாய்வு செய்து “Create” என்பதைக் கிளிக் செய்யவும். Azure தேவையான வளங்களை அமைத்து, உங்கள் களஞ்சியத்தில் GitHub Actions வேலைப்பாட்டை உருவாக்கும்.  

5. GitHub Actions வேலைப்பாடு  
Azure உங்கள் களஞ்சியத்தில் (.github/workflows/azure-static-web-apps-<name>.yml) ஒரு GitHub Actions வேலைப்பாட்டு கோப்பை தானாக உருவாக்கும். இந்த வேலைப்பாடு உருவாக்கம் மற்றும் வெளியீட்டு செயல்முறையை கையாளும்.  

6. வெளியீட்டை கண்காணிக்கவும்  
உங்கள் GitHub களஞ்சியத்தில் “Actions” தாவலை நோக்கி செல்லவும்.  
ஒரு வேலைப்பாடு இயங்குவதை நீங்கள் காணலாம். இந்த வேலைப்பாடு உங்கள் static web app ஐ Azure-க்கு உருவாக்கி வெளியிடும்.  
வேலைப்பாடு முடிந்தவுடன், உங்கள் செயலி வழங்கப்பட்ட Azure URL இல் நேரடியாக இருக்கும்.  

### உதாரண வேலைப்பாட்டு கோப்பு  

இது GitHub Actions வேலைப்பாட்டு கோப்பின் உதாரணமாக இருக்கலாம்:  
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
  
### கூடுதல் வளங்கள்  
- [Azure Static Web Apps Documentation](https://learn.microsoft.com/azure/static-web-apps/getting-started)  
- [GitHub Actions Documentation](https://docs.github.com/actions/use-cases-and-examples/deploying/deploying-to-azure-static-web-app)  

---

**குறிப்பு**:  
இந்த ஆவணம் [Co-op Translator](https://github.com/Azure/co-op-translator) என்ற AI மொழிபெயர்ப்பு சேவையைப் பயன்படுத்தி மொழிபெயர்க்கப்பட்டுள்ளது. நாங்கள் துல்லியத்திற்காக முயற்சிக்கின்றோம், ஆனால் தானியங்கி மொழிபெயர்ப்புகளில் பிழைகள் அல்லது தவறான தகவல்கள் இருக்கக்கூடும் என்பதை கவனத்தில் கொள்ளவும். அதன் தாய்மொழியில் உள்ள மூல ஆவணம் அதிகாரப்பூர்வ ஆதாரமாக கருதப்பட வேண்டும். முக்கியமான தகவல்களுக்கு, தொழில்முறை மனித மொழிபெயர்ப்பு பரிந்துரைக்கப்படுகிறது. இந்த மொழிபெயர்ப்பைப் பயன்படுத்துவதால் ஏற்படும் எந்த தவறான புரிதல்கள் அல்லது தவறான விளக்கங்களுக்கு நாங்கள் பொறுப்பல்ல.