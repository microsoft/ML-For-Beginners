<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "6d130dffca5db70d7e615f926cb1ad4c",
  "translation_date": "2025-09-03T23:48:23+00:00",
  "source_file": "quiz-app/README.md",
  "language_code": "fa"
}
-->
# آزمون‌ها

این آزمون‌ها، آزمون‌های قبل و بعد از جلسات آموزشی برای برنامه درسی یادگیری ماشین در https://aka.ms/ml-beginners هستند.

## تنظیم پروژه

```
npm install
```

### کامپایل و بارگذاری مجدد برای توسعه

```
npm run serve
```

### کامپایل و کوچک‌سازی برای تولید

```
npm run build
```

### بررسی و اصلاح فایل‌ها

```
npm run lint
```

### سفارشی‌سازی تنظیمات

به [مرجع تنظیمات](https://cli.vuejs.org/config/) مراجعه کنید.

اعتبار: سپاس از نسخه اصلی این اپلیکیشن آزمون: https://github.com/arpan45/simple-quiz-vue

## استقرار در Azure

در اینجا یک راهنمای گام‌به‌گام برای شروع کار آورده شده است:

1. فورک کردن یک مخزن GitHub  
اطمینان حاصل کنید که کد اپلیکیشن وب استاتیک شما در مخزن GitHub شما قرار دارد. این مخزن را فورک کنید.

2. ایجاد یک اپلیکیشن وب استاتیک Azure  
- یک [حساب Azure](http://azure.microsoft.com) ایجاد کنید.  
- به [پرتال Azure](https://portal.azure.com) بروید.  
- روی "ایجاد یک منبع" کلیک کنید و "Static Web App" را جستجو کنید.  
- روی "ایجاد" کلیک کنید.  

3. پیکربندی اپلیکیشن وب استاتیک  
- اصول:  
  - اشتراک: اشتراک Azure خود را انتخاب کنید.  
  - گروه منابع: یک گروه منابع جدید ایجاد کنید یا از یک گروه موجود استفاده کنید.  
  - نام: یک نام برای اپلیکیشن وب استاتیک خود ارائه دهید.  
  - منطقه: منطقه‌ای را انتخاب کنید که به کاربران شما نزدیک‌تر باشد.  

- #### جزئیات استقرار:  
  - منبع: "GitHub" را انتخاب کنید.  
  - حساب GitHub: به Azure اجازه دهید به حساب GitHub شما دسترسی پیدا کند.  
  - سازمان: سازمان GitHub خود را انتخاب کنید.  
  - مخزن: مخزنی را که شامل اپلیکیشن وب استاتیک شما است انتخاب کنید.  
  - شاخه: شاخه‌ای را که می‌خواهید از آن استقرار انجام دهید انتخاب کنید.  

- #### جزئیات ساخت:  
  - پیش‌تنظیم‌های ساخت: چارچوبی را که اپلیکیشن شما با آن ساخته شده است انتخاب کنید (مانند React، Angular، Vue و غیره).  
  - مکان اپلیکیشن: پوشه‌ای را که شامل کد اپلیکیشن شما است مشخص کنید (مانند / اگر در ریشه باشد).  
  - مکان API: اگر API دارید، مکان آن را مشخص کنید (اختیاری).  
  - مکان خروجی: پوشه‌ای را که خروجی ساخت در آن تولید می‌شود مشخص کنید (مانند build یا dist).  

4. بررسی و ایجاد  
تنظیمات خود را بررسی کنید و روی "ایجاد" کلیک کنید. Azure منابع لازم را تنظیم کرده و یک فایل گردش کار GitHub Actions در مخزن شما ایجاد خواهد کرد.

5. گردش کار GitHub Actions  
Azure به طور خودکار یک فایل گردش کار GitHub Actions در مخزن شما ایجاد می‌کند (.github/workflows/azure-static-web-apps-<name>.yml). این گردش کار فرآیند ساخت و استقرار را مدیریت خواهد کرد.

6. نظارت بر استقرار  
به تب "Actions" در مخزن GitHub خود بروید.  
باید یک گردش کار در حال اجرا را مشاهده کنید. این گردش کار اپلیکیشن وب استاتیک شما را در Azure می‌سازد و مستقر می‌کند.  
پس از تکمیل گردش کار، اپلیکیشن شما در URL ارائه شده Azure فعال خواهد بود.

### فایل نمونه گردش کار

در اینجا یک نمونه از فایل گردش کار GitHub Actions آورده شده است:  
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

### منابع اضافی  
- [مستندات اپلیکیشن‌های وب استاتیک Azure](https://learn.microsoft.com/azure/static-web-apps/getting-started)  
- [مستندات GitHub Actions](https://docs.github.com/actions/use-cases-and-examples/deploying/deploying-to-azure-static-web-app)  

---

**سلب مسئولیت**:  
این سند با استفاده از سرویس ترجمه هوش مصنوعی [Co-op Translator](https://github.com/Azure/co-op-translator) ترجمه شده است. در حالی که ما تلاش می‌کنیم دقت را حفظ کنیم، لطفاً توجه داشته باشید که ترجمه‌های خودکار ممکن است شامل خطاها یا نادرستی‌ها باشند. سند اصلی به زبان اصلی آن باید به عنوان منبع معتبر در نظر گرفته شود. برای اطلاعات حساس، توصیه می‌شود از ترجمه حرفه‌ای انسانی استفاده کنید. ما مسئولیتی در قبال سوءتفاهم‌ها یا تفسیرهای نادرست ناشی از استفاده از این ترجمه نداریم.