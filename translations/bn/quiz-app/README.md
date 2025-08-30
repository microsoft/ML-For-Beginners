<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "6d130dffca5db70d7e615f926cb1ad4c",
  "translation_date": "2025-08-29T21:38:22+00:00",
  "source_file": "quiz-app/README.md",
  "language_code": "bn"
}
-->
# কুইজ

এই কুইজগুলো ML কারিকুলামের প্রাক-লেকচার এবং পোস্ট-লেকচার কুইজ, যা পাওয়া যাবে https://aka.ms/ml-beginners এ।

## প্রজেক্ট সেটআপ

```
npm install
```

### ডেভেলপমেন্টের জন্য কম্পাইল এবং হট-রিলোড

```
npm run serve
```

### প্রোডাকশনের জন্য কম্পাইল এবং মিনিফাই

```
npm run build
```

### ফাইল লিন্ট এবং ফিক্স

```
npm run lint
```

### কনফিগারেশন কাস্টমাইজ করুন

[Configuration Reference](https://cli.vuejs.org/config/) দেখুন।

ক্রেডিট: এই কুইজ অ্যাপের মূল সংস্করণের জন্য ধন্যবাদ: https://github.com/arpan45/simple-quiz-vue

## Azure-এ ডিপ্লয় করা

এখানে একটি ধাপে ধাপে গাইড দেওয়া হলো যা আপনাকে শুরু করতে সাহায্য করবে:

1. একটি GitHub রিপোজিটরি ফর্ক করুন  
আপনার স্ট্যাটিক ওয়েব অ্যাপ কোডটি আপনার GitHub রিপোজিটরিতে থাকতে হবে। এই রিপোজিটরিটি ফর্ক করুন।

2. একটি Azure স্ট্যাটিক ওয়েব অ্যাপ তৈরি করুন  
- একটি [Azure অ্যাকাউন্ট](http://azure.microsoft.com) তৈরি করুন  
- [Azure পোর্টাল](https://portal.azure.com) এ যান  
- “Create a resource” এ ক্লিক করুন এবং “Static Web App” অনুসন্ধান করুন।  
- “Create” এ ক্লিক করুন।  

3. স্ট্যাটিক ওয়েব অ্যাপ কনফিগার করুন  
- #### বেসিকস:  
  - Subscription: আপনার Azure সাবস্ক্রিপশন নির্বাচন করুন।  
  - Resource Group: একটি নতুন রিসোর্স গ্রুপ তৈরি করুন অথবা বিদ্যমানটি ব্যবহার করুন।  
  - Name: আপনার স্ট্যাটিক ওয়েব অ্যাপের জন্য একটি নাম দিন।  
  - Region: আপনার ব্যবহারকারীদের কাছাকাছি অঞ্চল নির্বাচন করুন।  

- #### ডিপ্লয়মেন্ট ডিটেইলস:  
  - Source: “GitHub” নির্বাচন করুন।  
  - GitHub Account: Azure-কে আপনার GitHub অ্যাকাউন্টে অ্যাক্সেস করার অনুমতি দিন।  
  - Organization: আপনার GitHub অর্গানাইজেশন নির্বাচন করুন।  
  - Repository: আপনার স্ট্যাটিক ওয়েব অ্যাপের রিপোজিটরি নির্বাচন করুন।  
  - Branch: যে ব্রাঞ্চ থেকে ডিপ্লয় করতে চান তা নির্বাচন করুন।  

- #### বিল্ড ডিটেইলস:  
  - Build Presets: আপনার অ্যাপটি যে ফ্রেমওয়ার্ক দিয়ে তৈরি (যেমন React, Angular, Vue ইত্যাদি) তা নির্বাচন করুন।  
  - App Location: আপনার অ্যাপ কোডের অবস্থান নির্ধারণ করুন (যেমন, / যদি এটি রুটে থাকে)।  
  - API Location: যদি আপনার API থাকে, তার অবস্থান নির্ধারণ করুন (ঐচ্ছিক)।  
  - Output Location: যেখানে বিল্ড আউটপুট তৈরি হয় সেই ফোল্ডার নির্ধারণ করুন (যেমন, build বা dist)।  

4. রিভিউ এবং তৈরি করুন  
আপনার সেটিংস রিভিউ করুন এবং “Create” এ ক্লিক করুন। Azure প্রয়োজনীয় রিসোর্স সেটআপ করবে এবং আপনার রিপোজিটরিতে একটি GitHub Actions ওয়ার্কফ্লো তৈরি করবে।  

5. GitHub Actions ওয়ার্কফ্লো  
Azure স্বয়ংক্রিয়ভাবে আপনার রিপোজিটরিতে একটি GitHub Actions ওয়ার্কফ্লো ফাইল তৈরি করবে (.github/workflows/azure-static-web-apps-<name>.yml)। এই ওয়ার্কফ্লো বিল্ড এবং ডিপ্লয়মেন্ট প্রক্রিয়া পরিচালনা করবে।  

6. ডিপ্লয়মেন্ট মনিটর করুন  
আপনার GitHub রিপোজিটরির “Actions” ট্যাবে যান।  
আপনার একটি ওয়ার্কফ্লো চলমান দেখতে পাবেন। এই ওয়ার্কফ্লো আপনার স্ট্যাটিক ওয়েব অ্যাপকে Azure-এ বিল্ড এবং ডিপ্লয় করবে।  
ওয়ার্কফ্লো সম্পন্ন হলে, আপনার অ্যাপটি প্রদত্ত Azure URL-এ লাইভ হবে।  

### উদাহরণ ওয়ার্কফ্লো ফাইল

এখানে GitHub Actions ওয়ার্কফ্লো ফাইলের একটি উদাহরণ দেওয়া হলো:  
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

### অতিরিক্ত রিসোর্স  
- [Azure Static Web Apps Documentation](https://learn.microsoft.com/azure/static-web-apps/getting-started)  
- [GitHub Actions Documentation](https://docs.github.com/actions/use-cases-and-examples/deploying/deploying-to-azure-static-web-app)  

---

**অস্বীকৃতি**:  
এই নথিটি AI অনুবাদ পরিষেবা [Co-op Translator](https://github.com/Azure/co-op-translator) ব্যবহার করে অনুবাদ করা হয়েছে। আমরা যথাসম্ভব সঠিক অনুবাদ প্রদানের চেষ্টা করি, তবে অনুগ্রহ করে মনে রাখবেন যে স্বয়ংক্রিয় অনুবাদে ত্রুটি বা অসঙ্গতি থাকতে পারে। মূল ভাষায় থাকা নথিটিকে প্রামাণিক উৎস হিসেবে বিবেচনা করা উচিত। গুরুত্বপূর্ণ তথ্যের জন্য, পেশাদার মানব অনুবাদ সুপারিশ করা হয়। এই অনুবাদ ব্যবহারের ফলে কোনো ভুল বোঝাবুঝি বা ভুল ব্যাখ্যা হলে আমরা তার জন্য দায়ী থাকব না।