<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "93fdaa0fd38836e50c4793e2f2f25e8b",
  "translation_date": "2025-10-03T11:09:19+00:00",
  "source_file": "AGENTS.md",
  "language_code": "th"
}
-->
# AGENTS.md

## ภาพรวมของโครงการ

นี่คือ **Machine Learning for Beginners** หลักสูตรครอบคลุม 12 สัปดาห์ 26 บทเรียนที่เน้นแนวคิดการเรียนรู้ของเครื่องแบบคลาสสิกโดยใช้ Python (ส่วนใหญ่ใช้ Scikit-learn) และ R โครงการนี้ออกแบบมาเพื่อการเรียนรู้ด้วยตนเอง พร้อมด้วยโปรเจกต์แบบลงมือทำ คำถามแบบทดสอบ และงานมอบหมาย แต่ละบทเรียนจะสำรวจแนวคิด ML ผ่านข้อมูลจริงจากวัฒนธรรมและภูมิภาคต่างๆ ทั่วโลก

องค์ประกอบสำคัญ:
- **เนื้อหาการศึกษา**: 26 บทเรียนครอบคลุมการแนะนำ ML, การถดถอย, การจำแนกประเภท, การจัดกลุ่ม, NLP, การวิเคราะห์อนุกรมเวลา และการเรียนรู้แบบเสริมแรง
- **แอปพลิเคชันแบบทดสอบ**: แอปแบบทดสอบที่สร้างด้วย Vue.js พร้อมการประเมินก่อนและหลังบทเรียน
- **รองรับหลายภาษา**: การแปลอัตโนมัติเป็นมากกว่า 40 ภาษาโดยใช้ GitHub Actions
- **รองรับสองภาษา**: บทเรียนมีทั้งใน Python (Jupyter notebooks) และ R (ไฟล์ R Markdown)
- **การเรียนรู้แบบโปรเจกต์**: แต่ละหัวข้อมีโปรเจกต์และงานมอบหมายที่ลงมือทำจริง

## โครงสร้างของ Repository

```
ML-For-Beginners/
├── 1-Introduction/         # ML basics, history, fairness, techniques
├── 2-Regression/          # Regression models with Python/R
├── 3-Web-App/            # Flask web app for ML model deployment
├── 4-Classification/      # Classification algorithms
├── 5-Clustering/         # Clustering techniques
├── 6-NLP/               # Natural Language Processing
├── 7-TimeSeries/        # Time series forecasting
├── 8-Reinforcement/     # Reinforcement learning
├── 9-Real-World/        # Real-world ML applications
├── quiz-app/           # Vue.js quiz application
├── translations/       # Auto-generated translations
└── sketchnotes/       # Visual learning aids
```

แต่ละโฟลเดอร์บทเรียนมักจะมี:
- `README.md` - เนื้อหาหลักของบทเรียน
- `notebook.ipynb` - Jupyter notebook สำหรับ Python
- `solution/` - โค้ดคำตอบ (เวอร์ชัน Python และ R)
- `assignment.md` - แบบฝึกหัด
- `images/` - ทรัพยากรภาพประกอบ

## คำสั่งการตั้งค่า

### สำหรับบทเรียน Python

บทเรียนส่วนใหญ่ใช้ Jupyter notebooks ติดตั้ง dependencies ที่จำเป็น:

```bash
# Install Python 3.8+ if not already installed
python --version

# Install Jupyter
pip install jupyter

# Install common ML libraries
pip install scikit-learn pandas numpy matplotlib seaborn

# For specific lessons, check lesson-specific requirements
# Example: Web App lesson
pip install flask
```

### สำหรับบทเรียน R

บทเรียน R อยู่ในโฟลเดอร์ `solution/R/` ในรูปแบบ `.rmd` หรือ `.ipynb`:

```bash
# Install R and required packages
# In R console:
install.packages(c("tidyverse", "tidymodels", "caret"))
```

### สำหรับแอปพลิเคชันแบบทดสอบ

แอปแบบทดสอบเป็นแอป Vue.js ที่อยู่ในไดเรกทอรี `quiz-app/`:

```bash
cd quiz-app
npm install
```

### สำหรับเว็บไซต์เอกสาร

เพื่อรันเอกสารในเครื่อง:

```bash
# Install Docsify
npm install -g docsify-cli

# Serve from repository root
docsify serve

# Access at http://localhost:3000
```

## เวิร์กโฟลว์การพัฒนา

### การทำงานกับ Lesson Notebooks

1. ไปยังไดเรกทอรีบทเรียน (เช่น `2-Regression/1-Tools/`)
2. เปิด Jupyter notebook:
   ```bash
   jupyter notebook notebook.ipynb
   ```
3. ทำเนื้อหาและแบบฝึกหัดในบทเรียน
4. ตรวจสอบคำตอบในโฟลเดอร์ `solution/` หากจำเป็น

### การพัฒนา Python

- บทเรียนใช้ไลบรารีข้อมูลมาตรฐานของ Python
- Jupyter notebooks สำหรับการเรียนรู้แบบโต้ตอบ
- โค้ดคำตอบมีอยู่ในโฟลเดอร์ `solution/` ของแต่ละบทเรียน

### การพัฒนา R

- บทเรียน R อยู่ในรูปแบบ `.rmd` (R Markdown)
- คำตอบอยู่ในโฟลเดอร์ย่อย `solution/R/`
- ใช้ RStudio หรือ Jupyter พร้อม R kernel เพื่อรัน R notebooks

### การพัฒนาแอปพลิเคชันแบบทดสอบ

```bash
cd quiz-app

# Start development server
npm run serve
# Access at http://localhost:8080

# Build for production
npm run build

# Lint and fix files
npm run lint
```

## คำแนะนำการทดสอบ

### การทดสอบแอปพลิเคชันแบบทดสอบ

```bash
cd quiz-app

# Lint code
npm run lint

# Build to verify no errors
npm run build
```

**หมายเหตุ**: นี่เป็น repository หลักสูตรการศึกษา ไม่มีการทดสอบอัตโนมัติสำหรับเนื้อหาบทเรียน การตรวจสอบทำได้โดย:
- ทำแบบฝึกหัดในบทเรียน
- รันเซลล์ใน notebook สำเร็จ
- ตรวจสอบผลลัพธ์กับคำตอบที่คาดหวังในโฟลเดอร์คำตอบ

## แนวทางการเขียนโค้ด

### โค้ด Python
- ปฏิบัติตามแนวทาง PEP 8
- ใช้ชื่อตัวแปรที่ชัดเจนและอธิบายได้
- เพิ่มคอมเมนต์สำหรับการดำเนินการที่ซับซ้อน
- Jupyter notebooks ควรมีเซลล์ markdown อธิบายแนวคิด

### JavaScript/Vue.js (แอปแบบทดสอบ)
- ปฏิบัติตามแนวทางของ Vue.js
- การตั้งค่า ESLint ใน `quiz-app/package.json`
- รัน `npm run lint` เพื่อตรวจสอบและแก้ไขปัญหาอัตโนมัติ

### เอกสาร
- ไฟล์ Markdown ควรชัดเจนและมีโครงสร้างดี
- รวมตัวอย่างโค้ดใน fenced code blocks
- ใช้ลิงก์แบบสัมพัทธ์สำหรับการอ้างอิงภายใน
- ปฏิบัติตามรูปแบบที่มีอยู่แล้ว

## การสร้างและการปรับใช้

### การปรับใช้แอปพลิเคชันแบบทดสอบ

แอปแบบทดสอบสามารถปรับใช้ใน Azure Static Web Apps:

1. **ข้อกำหนดเบื้องต้น**:
   - บัญชี Azure
   - Repository GitHub (ที่ fork แล้ว)

2. **ปรับใช้ใน Azure**:
   - สร้างทรัพยากร Azure Static Web App
   - เชื่อมต่อกับ Repository GitHub
   - ตั้งค่าตำแหน่งแอป: `/quiz-app`
   - ตั้งค่าตำแหน่งผลลัพธ์: `dist`
   - Azure สร้าง GitHub Actions workflow อัตโนมัติ

3. **GitHub Actions Workflow**:
   - ไฟล์ workflow สร้างที่ `.github/workflows/azure-static-web-apps-*.yml`
   - สร้างและปรับใช้อัตโนมัติเมื่อ push ไปยัง branch main

### เอกสาร PDF

สร้าง PDF จากเอกสาร:

```bash
npm install
npm run convert
```

## เวิร์กโฟลว์การแปลภาษา

**สำคัญ**: การแปลภาษาเป็นแบบอัตโนมัติผ่าน GitHub Actions โดยใช้ Co-op Translator

- การแปลจะถูกสร้างอัตโนมัติเมื่อมีการเปลี่ยนแปลง push ไปยัง branch `main`
- **ห้ามแปลเนื้อหาด้วยตนเอง** - ระบบจัดการให้
- Workflow กำหนดใน `.github/workflows/co-op-translator.yml`
- ใช้บริการ Azure AI/OpenAI สำหรับการแปล
- รองรับมากกว่า 40 ภาษา

## แนวทางการมีส่วนร่วม

### สำหรับผู้มีส่วนร่วมด้านเนื้อหา

1. **Fork repository** และสร้าง feature branch
2. **แก้ไขเนื้อหาบทเรียน** หากเพิ่ม/อัปเดตบทเรียน
3. **ห้ามแก้ไขไฟล์ที่แปลแล้ว** - ระบบสร้างอัตโนมัติ
4. **ทดสอบโค้ดของคุณ** - ตรวจสอบให้แน่ใจว่าเซลล์ใน notebook ทั้งหมดรันสำเร็จ
5. **ตรวจสอบลิงก์และภาพ** ว่าทำงานถูกต้อง
6. **ส่ง pull request** พร้อมคำอธิบายที่ชัดเจน

### แนวทางการส่ง Pull Request

- **รูปแบบชื่อเรื่อง**: `[Section] คำอธิบายการเปลี่ยนแปลงโดยย่อ`
  - ตัวอย่าง: `[Regression] แก้ไขคำผิดในบทเรียน 5`
  - ตัวอย่าง: `[Quiz-App] อัปเดต dependencies`
- **ก่อนส่ง**:
  - ตรวจสอบให้แน่ใจว่าเซลล์ใน notebook ทั้งหมดรันโดยไม่มีข้อผิดพลาด
  - รัน `npm run lint` หากแก้ไข quiz-app
  - ตรวจสอบรูปแบบ Markdown
  - ทดสอบตัวอย่างโค้ดใหม่
- **PR ต้องมี**:
  - คำอธิบายการเปลี่ยนแปลง
  - เหตุผลของการเปลี่ยนแปลง
  - ภาพหน้าจอหากมีการเปลี่ยนแปลง UI
- **จรรยาบรรณ**: ปฏิบัติตาม [Microsoft Open Source Code of Conduct](CODE_OF_CONDUCT.md)
- **CLA**: คุณจะต้องลงนามใน Contributor License Agreement

## โครงสร้างบทเรียน

แต่ละบทเรียนมีรูปแบบที่สม่ำเสมอ:

1. **แบบทดสอบก่อนการบรรยาย** - ทดสอบความรู้พื้นฐาน
2. **เนื้อหาบทเรียน** - คำแนะนำและคำอธิบายที่เขียนไว้
3. **การสาธิตโค้ด** - ตัวอย่างแบบลงมือทำใน notebooks
4. **การตรวจสอบความรู้** - ตรวจสอบความเข้าใจตลอดบทเรียน
5. **ความท้าทาย** - ใช้แนวคิดด้วยตนเอง
6. **งานมอบหมาย** - การฝึกฝนเพิ่มเติม
7. **แบบทดสอบหลังการบรรยาย** - ประเมินผลการเรียนรู้

## อ้างอิงคำสั่งทั่วไป

```bash
# Python/Jupyter
jupyter notebook                    # Start Jupyter server
jupyter notebook notebook.ipynb     # Open specific notebook
pip install -r requirements.txt     # Install dependencies (where available)

# Quiz App
cd quiz-app
npm install                        # Install dependencies
npm run serve                      # Development server
npm run build                      # Production build
npm run lint                       # Lint and fix

# Documentation
docsify serve                      # Serve documentation locally
npm run convert                    # Generate PDF

# Git workflow
git checkout -b feature/my-change  # Create feature branch
git add .                         # Stage changes
git commit -m "Description"       # Commit changes
git push origin feature/my-change # Push to remote
```

## ทรัพยากรเพิ่มเติม

- **Microsoft Learn Collection**: [โมดูล ML สำหรับผู้เริ่มต้น](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)
- **แอปแบบทดสอบ**: [แบบทดสอบออนไลน์](https://ff-quizzes.netlify.app/en/ml/)
- **กระดานสนทนา**: [GitHub Discussions](https://github.com/microsoft/ML-For-Beginners/discussions)
- **วิดีโอแนะนำ**: [YouTube Playlist](https://aka.ms/ml-beginners-videos)

## เทคโนโลยีสำคัญ

- **Python**: ภาษาเริ่มต้นสำหรับบทเรียน ML (Scikit-learn, Pandas, NumPy, Matplotlib)
- **R**: การใช้งานทางเลือกโดยใช้ tidyverse, tidymodels, caret
- **Jupyter**: Notebooks แบบโต้ตอบสำหรับบทเรียน Python
- **R Markdown**: เอกสารสำหรับบทเรียน R
- **Vue.js 3**: เฟรมเวิร์กสำหรับแอปแบบทดสอบ
- **Flask**: เฟรมเวิร์กเว็บแอปพลิเคชันสำหรับการปรับใช้โมเดล ML
- **Docsify**: ตัวสร้างเว็บไซต์เอกสาร
- **GitHub Actions**: CI/CD และการแปลอัตโนมัติ

## ข้อควรพิจารณาด้านความปลอดภัย

- **ไม่มีข้อมูลลับในโค้ด**: ห้าม commit API keys หรือข้อมูลรับรอง
- **Dependencies**: อัปเดตแพ็กเกจ npm และ pip ให้ทันสมัย
- **ข้อมูลผู้ใช้**: ตัวอย่างเว็บแอป Flask มีการตรวจสอบข้อมูลเบื้องต้น
- **ข้อมูลที่ละเอียดอ่อน**: ชุดข้อมูลตัวอย่างเป็นข้อมูลสาธารณะและไม่ละเอียดอ่อน

## การแก้ไขปัญหา

### Jupyter Notebooks

- **ปัญหา Kernel**: รีสตาร์ท kernel หากเซลล์ค้าง: Kernel → Restart
- **ข้อผิดพลาดการนำเข้า**: ตรวจสอบให้แน่ใจว่าติดตั้งแพ็กเกจที่จำเป็นทั้งหมดด้วย pip
- **ปัญหาเส้นทาง**: รัน notebooks จากไดเรกทอรีที่มีไฟล์นั้น

### แอปพลิเคชันแบบทดสอบ

- **npm install ล้มเหลว**: ล้างแคช npm: `npm cache clean --force`
- **ปัญหาพอร์ต**: เปลี่ยนพอร์ตด้วย: `npm run serve -- --port 8081`
- **ข้อผิดพลาดการสร้าง**: ลบ `node_modules` และติดตั้งใหม่: `rm -rf node_modules && npm install`

### บทเรียน R

- **แพ็กเกจไม่พบ**: ติดตั้งด้วย: `install.packages("package-name")`
- **การเรนเดอร์ RMarkdown**: ตรวจสอบให้แน่ใจว่าติดตั้งแพ็กเกจ rmarkdown
- **ปัญหา Kernel**: อาจต้องติดตั้ง IRkernel สำหรับ Jupyter

## หมายเหตุเฉพาะโครงการ

- นี่เป็นหลักสูตร **การเรียนรู้** ไม่ใช่โค้ดสำหรับการผลิต
- เน้นที่ **การเข้าใจแนวคิด ML** ผ่านการลงมือทำ
- ตัวอย่างโค้ดให้ความสำคัญกับ **ความชัดเจนมากกว่าการปรับแต่ง**
- บทเรียนส่วนใหญ่ **เป็นแบบแยกส่วน** และสามารถทำได้อย่างอิสระ
- **มีคำตอบให้** แต่ผู้เรียนควรลองทำแบบฝึกหัดก่อน
- Repository ใช้ **Docsify** สำหรับเอกสารเว็บโดยไม่ต้องมีขั้นตอนการสร้าง
- **Sketchnotes** ให้สรุปภาพรวมของแนวคิด
- **รองรับหลายภาษา** ทำให้เนื้อหาเข้าถึงได้ทั่วโลก

---

**ข้อจำกัดความรับผิดชอบ**:  
เอกสารนี้ได้รับการแปลโดยใช้บริการแปลภาษา AI [Co-op Translator](https://github.com/Azure/co-op-translator) แม้ว่าเราจะพยายามให้การแปลมีความถูกต้อง แต่โปรดทราบว่าการแปลอัตโนมัติอาจมีข้อผิดพลาดหรือความไม่ถูกต้อง เอกสารต้นฉบับในภาษาดั้งเดิมควรถือเป็นแหล่งข้อมูลที่เชื่อถือได้ สำหรับข้อมูลที่สำคัญ ขอแนะนำให้ใช้บริการแปลภาษามนุษย์ที่มีความเชี่ยวชาญ เราไม่รับผิดชอบต่อความเข้าใจผิดหรือการตีความผิดที่เกิดจากการใช้การแปลนี้