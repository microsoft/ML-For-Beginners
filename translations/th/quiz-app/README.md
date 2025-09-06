<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "6d130dffca5db70d7e615f926cb1ad4c",
  "translation_date": "2025-09-05T21:48:20+00:00",
  "source_file": "quiz-app/README.md",
  "language_code": "th"
}
-->
# แบบทดสอบ

แบบทดสอบเหล่านี้เป็นแบบทดสอบก่อนและหลังการบรรยายสำหรับหลักสูตร ML ที่ https://aka.ms/ml-beginners

## การตั้งค่าโปรเจกต์

```
npm install
```

### คอมไพล์และรีโหลดแบบสดสำหรับการพัฒนา

```
npm run serve
```

### คอมไพล์และย่อขนาดสำหรับการผลิต

```
npm run build
```

### ตรวจสอบและแก้ไขไฟล์

```
npm run lint
```

### ปรับแต่งการตั้งค่า

ดู [การอ้างอิงการตั้งค่า](https://cli.vuejs.org/config/).

เครดิต: ขอบคุณเวอร์ชันต้นฉบับของแอปแบบทดสอบนี้: https://github.com/arpan45/simple-quiz-vue

## การปรับใช้บน Azure

นี่คือคำแนะนำทีละขั้นตอนเพื่อช่วยให้คุณเริ่มต้นได้:

1. Fork GitHub Repository  
ตรวจสอบให้แน่ใจว่าโค้ดเว็บแอปแบบสแตติกของคุณอยู่ใน GitHub repository ของคุณ Fork repository นี้

2. สร้าง Azure Static Web App  
- สร้าง [บัญชี Azure](http://azure.microsoft.com)  
- ไปที่ [Azure portal](https://portal.azure.com)  
- คลิกที่ “Create a resource” และค้นหา “Static Web App”  
- คลิก “Create”  

3. ตั้งค่า Static Web App  
- พื้นฐาน:  
  - Subscription: เลือกการสมัครใช้งาน Azure ของคุณ  
  - Resource Group: สร้าง resource group ใหม่หรือใช้ resource group ที่มีอยู่  
  - Name: ตั้งชื่อสำหรับเว็บแอปแบบสแตติกของคุณ  
  - Region: เลือกภูมิภาคที่ใกล้กับผู้ใช้ของคุณ  

- #### รายละเอียดการปรับใช้:  
  - Source: เลือก “GitHub”  
  - GitHub Account: อนุญาตให้ Azure เข้าถึงบัญชี GitHub ของคุณ  
  - Organization: เลือกองค์กร GitHub ของคุณ  
  - Repository: เลือก repository ที่มีเว็บแอปแบบสแตติกของคุณ  
  - Branch: เลือก branch ที่คุณต้องการปรับใช้  

- #### รายละเอียดการสร้าง:  
  - Build Presets: เลือก framework ที่แอปของคุณสร้างขึ้น (เช่น React, Angular, Vue เป็นต้น)  
  - App Location: ระบุโฟลเดอร์ที่มีโค้ดแอปของคุณ (เช่น / หากอยู่ใน root)  
  - API Location: หากคุณมี API ให้ระบุตำแหน่ง (ไม่บังคับ)  
  - Output Location: ระบุโฟลเดอร์ที่สร้างผลลัพธ์การ build (เช่น build หรือ dist)  

4. ตรวจสอบและสร้าง  
ตรวจสอบการตั้งค่าของคุณและคลิก “Create” Azure จะตั้งค่าทรัพยากรที่จำเป็นและสร้าง GitHub Actions workflow ใน repository ของคุณ  

5. GitHub Actions Workflow  
Azure จะสร้างไฟล์ GitHub Actions workflow ใน repository ของคุณโดยอัตโนมัติ (.github/workflows/azure-static-web-apps-<name>.yml) ไฟล์ workflow นี้จะจัดการกระบวนการ build และปรับใช้  

6. ตรวจสอบการปรับใช้  
ไปที่แท็บ “Actions” ใน GitHub repository ของคุณ  
คุณควรเห็น workflow กำลังทำงาน Workflow นี้จะ build และปรับใช้เว็บแอปแบบสแตติกของคุณไปยัง Azure  
เมื่อ workflow เสร็จสิ้น แอปของคุณจะใช้งานได้บน URL ที่ Azure ให้มา  

### ตัวอย่างไฟล์ Workflow  

นี่คือตัวอย่างไฟล์ GitHub Actions workflow:  
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

### แหล่งข้อมูลเพิ่มเติม  
- [เอกสาร Azure Static Web Apps](https://learn.microsoft.com/azure/static-web-apps/getting-started)  
- [เอกสาร GitHub Actions](https://docs.github.com/actions/use-cases-and-examples/deploying/deploying-to-azure-static-web-app)  

---

**ข้อจำกัดความรับผิดชอบ**:  
เอกสารนี้ได้รับการแปลโดยใช้บริการแปลภาษา AI [Co-op Translator](https://github.com/Azure/co-op-translator) แม้ว่าเราจะพยายามให้การแปลมีความถูกต้อง แต่โปรดทราบว่าการแปลอัตโนมัติอาจมีข้อผิดพลาดหรือความไม่ถูกต้อง เอกสารต้นฉบับในภาษาดั้งเดิมควรถือเป็นแหล่งข้อมูลที่เชื่อถือได้ สำหรับข้อมูลที่สำคัญ ขอแนะนำให้ใช้บริการแปลภาษามืออาชีพ เราไม่รับผิดชอบต่อความเข้าใจผิดหรือการตีความผิดที่เกิดจากการใช้การแปลนี้