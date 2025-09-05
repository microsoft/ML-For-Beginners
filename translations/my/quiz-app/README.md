<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "6d130dffca5db70d7e615f926cb1ad4c",
  "translation_date": "2025-09-05T13:02:37+00:00",
  "source_file": "quiz-app/README.md",
  "language_code": "my"
}
-->
# မေးခွန်းများ

ဒီမေးခွန်းများက ML သင်ခန်းစာများအတွက် [https://aka.ms/ml-beginners](https://aka.ms/ml-beginners) မှာရှိတဲ့ သင်ခန်းစာမတိုင်မီနှင့် သင်ခန်းစာပြီးနောက် မေးခွန်းများဖြစ်ပါတယ်။

## Project setup

```
npm install
```

### Development အတွက် Compile လုပ်ပြီး Hot-reload

```
npm run serve
```

### Production အတွက် Compile လုပ်ပြီး Minify

```
npm run build
```

### ဖိုင်များကို Lint လုပ်ပြီး ပြင်ဆင်

```
npm run lint
```

### Configuration ကို Customize လုပ်ရန်

[Configuration Reference](https://cli.vuejs.org/config/) ကိုကြည့်ပါ။

Credit: ဒီ Quiz App ရဲ့ မူရင်းဗားရှင်းကို ဖန်တီးသူ [https://github.com/arpan45/simple-quiz-vue](https://github.com/arpan45/simple-quiz-vue) ကို ကျေးဇူးတင်ပါတယ်။

## Azure မှာ Deploy လုပ်ခြင်း

ဒီအဆင့်ဆင့်လမ်းညွှန်ကို အသုံးပြုပြီး စတင်လုပ်ဆောင်ပါ:

1. GitHub Repository ကို Fork လုပ်ပါ  
သင့် Static Web App Code ကို GitHub Repository မှာရှိအောင်လုပ်ပါ။ ဒီ Repository ကို Fork လုပ်ပါ။

2. Azure Static Web App တစ်ခု Create လုပ်ပါ  
- [Azure account](http://azure.microsoft.com) တစ်ခု Create လုပ်ပါ  
- [Azure portal](https://portal.azure.com) ကိုသွားပါ  
- “Create a resource” ကိုနှိပ်ပြီး “Static Web App” ကို ရှာပါ။  
- “Create” ကိုနှိပ်ပါ။

3. Static Web App ကို Configure လုပ်ပါ  
- #### Basics:  
  - Subscription: သင့် Azure subscription ကို ရွေးပါ။  
  - Resource Group: Resource group အသစ်တစ်ခု Create လုပ်ပါ၊ ဒါမှမဟုတ် ရှိပြီးသား Resource group ကို အသုံးပြုပါ။  
  - Name: သင့် Static Web App အတွက် နာမည်ပေးပါ။  
  - Region: သင့်အသုံးပြုသူများနီးစပ်ရာ Region ကို ရွေးပါ။  

- #### Deployment Details:  
  - Source: “GitHub” ကို ရွေးပါ။  
  - GitHub Account: Azure ကို သင့် GitHub account ကို အသုံးပြုခွင့်ပေးပါ။  
  - Organization: သင့် GitHub organization ကို ရွေးပါ။  
  - Repository: သင့် Static Web App ရှိတဲ့ Repository ကို ရွေးပါ။  
  - Branch: Deploy လုပ်ချင်တဲ့ Branch ကို ရွေးပါ။  

- #### Build Details:  
  - Build Presets: သင့် App ဖန်တီးထားတဲ့ Framework ကို ရွေးပါ (ဥပမာ React, Angular, Vue စသည်တို့)။  
  - App Location: သင့် App Code ရှိတဲ့ Folder ကို ဖော်ပြပါ (ဥပမာ / သို့မဟုတ် Root မှာရှိပါက /)။  
  - API Location: API ရှိပါက အဲဒီနေရာကို ဖော်ပြပါ (optional)။  
  - Output Location: Build output ဖိုင်များ ရှိတဲ့ Folder ကို ဖော်ပြပါ (ဥပမာ build သို့မဟုတ် dist)။

4. Review and Create  
သင့် Settings များကို ပြန်လည်ကြည့်ရှုပြီး “Create” ကိုနှိပ်ပါ။ Azure က လိုအပ်တဲ့ Resources များကို Setup လုပ်ပြီး GitHub Actions Workflow ကို သင့် Repository မှာ Create လုပ်ပါမည်။

5. GitHub Actions Workflow  
Azure က GitHub Actions Workflow ဖိုင် (.github/workflows/azure-static-web-apps-<name>.yml) ကို သင့် Repository မှာ အလိုအလျောက် Create လုပ်ပါမည်။ ဒီ Workflow က Build နှင့် Deployment လုပ်ငန်းစဉ်ကို Handle လုပ်ပါမည်။

6. Deployment ကို Monitor လုပ်ပါ  
GitHub Repository ရဲ့ “Actions” tab ကိုသွားပါ။  
Workflow တစ်ခု Run ဖြစ်နေသည်ကို တွေ့ရပါမည်။ ဒီ Workflow က သင့် Static Web App ကို Azure မှာ Build နှင့် Deploy လုပ်ပါမည်။  
Workflow ပြီးဆုံးပြီးနောက် သင့် App ကို Azure URL မှာ Live ဖြစ်နေပါမည်။

### Example Workflow File

GitHub Actions Workflow ဖိုင်ရဲ့ ဥပမာကို အောက်မှာကြည့်ပါ:  
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

### အပိုဆောင်း Resources  
- [Azure Static Web Apps Documentation](https://learn.microsoft.com/azure/static-web-apps/getting-started)  
- [GitHub Actions Documentation](https://docs.github.com/actions/use-cases-and-examples/deploying/deploying-to-azure-static-web-app)  

---

**ဝက်ဘ်ဆိုက်မှတ်ချက်**:  
ဤစာရွက်စာတမ်းကို AI ဘာသာပြန်ဝန်ဆောင်မှု [Co-op Translator](https://github.com/Azure/co-op-translator) ကို အသုံးပြု၍ ဘာသာပြန်ထားပါသည်။ ကျွန်ုပ်တို့သည် တိကျမှန်ကန်မှုအတွက် ကြိုးစားနေသော်လည်း၊ အလိုအလျောက်ဘာသာပြန်ဆိုမှုများတွင် အမှားများ သို့မဟုတ် မမှန်ကန်မှုများ ပါဝင်နိုင်သည်ကို သတိပြုပါ။ မူရင်းစာရွက်စာတမ်းကို ၎င်း၏ မူလဘာသာစကားဖြင့် အာဏာတည်သောရင်းမြစ်အဖြစ် သတ်မှတ်သင့်ပါသည်။ အရေးကြီးသော အချက်အလက်များအတွက် လူက ဘာသာပြန်ဆိုမှုကို အသုံးပြုရန် အကြံပြုပါသည်။ ဤဘာသာပြန်ကို အသုံးပြုခြင်းမှ ဖြစ်ပေါ်လာသော နားလည်မှုမှားမှုများ သို့မဟုတ် အဓိပ္ပာယ်မှားမှုများအတွက် ကျွန်ုပ်တို့သည် တာဝန်မယူပါ။