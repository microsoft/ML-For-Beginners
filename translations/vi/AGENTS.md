<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "93fdaa0fd38836e50c4793e2f2f25e8b",
  "translation_date": "2025-10-03T11:12:57+00:00",
  "source_file": "AGENTS.md",
  "language_code": "vi"
}
-->
# AGENTS.md

## Tổng quan dự án

Đây là **Học máy dành cho người mới bắt đầu**, một chương trình học toàn diện kéo dài 12 tuần với 26 bài học, bao gồm các khái niệm học máy cổ điển sử dụng Python (chủ yếu với Scikit-learn) và R. Kho lưu trữ được thiết kế như một tài nguyên học tập tự học với các dự án thực hành, bài kiểm tra và bài tập. Mỗi bài học khám phá các khái niệm học máy thông qua dữ liệu thực tế từ nhiều nền văn hóa và khu vực trên toàn thế giới.

Các thành phần chính:
- **Nội dung giáo dục**: 26 bài học bao gồm giới thiệu về học máy, hồi quy, phân loại, phân cụm, NLP, chuỗi thời gian và học tăng cường
- **Ứng dụng kiểm tra**: Ứng dụng kiểm tra dựa trên Vue.js với đánh giá trước và sau bài học
- **Hỗ trợ đa ngôn ngữ**: Dịch tự động sang hơn 40 ngôn ngữ thông qua GitHub Actions
- **Hỗ trợ hai ngôn ngữ**: Bài học có sẵn bằng cả Python (Jupyter notebooks) và R (tệp R Markdown)
- **Học tập dựa trên dự án**: Mỗi chủ đề bao gồm các dự án thực hành và bài tập

## Cấu trúc kho lưu trữ

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

Mỗi thư mục bài học thường chứa:
- `README.md` - Nội dung chính của bài học
- `notebook.ipynb` - Jupyter notebook Python
- `solution/` - Mã giải pháp (phiên bản Python và R)
- `assignment.md` - Bài tập thực hành
- `images/` - Tài nguyên hình ảnh

## Lệnh thiết lập

### Đối với bài học Python

Hầu hết các bài học sử dụng Jupyter notebooks. Cài đặt các phụ thuộc cần thiết:

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

### Đối với bài học R

Các bài học R nằm trong thư mục `solution/R/` dưới dạng tệp `.rmd` hoặc `.ipynb`:

```bash
# Install R and required packages
# In R console:
install.packages(c("tidyverse", "tidymodels", "caret"))
```

### Đối với ứng dụng kiểm tra

Ứng dụng kiểm tra là một ứng dụng Vue.js nằm trong thư mục `quiz-app/`:

```bash
cd quiz-app
npm install
```

### Đối với trang tài liệu

Để chạy tài liệu cục bộ:

```bash
# Install Docsify
npm install -g docsify-cli

# Serve from repository root
docsify serve

# Access at http://localhost:3000
```

## Quy trình phát triển

### Làm việc với các notebook bài học

1. Điều hướng đến thư mục bài học (ví dụ: `2-Regression/1-Tools/`)
2. Mở Jupyter notebook:
   ```bash
   jupyter notebook notebook.ipynb
   ```
3. Làm việc qua nội dung bài học và bài tập
4. Kiểm tra giải pháp trong thư mục `solution/` nếu cần

### Phát triển Python

- Các bài học sử dụng thư viện khoa học dữ liệu tiêu chuẩn của Python
- Jupyter notebooks để học tương tác
- Mã giải pháp có sẵn trong thư mục `solution/` của mỗi bài học

### Phát triển R

- Các bài học R ở định dạng `.rmd` (R Markdown)
- Giải pháp nằm trong các thư mục con `solution/R/`
- Sử dụng RStudio hoặc Jupyter với kernel R để chạy các notebook R

### Phát triển ứng dụng kiểm tra

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

## Hướng dẫn kiểm tra

### Kiểm tra ứng dụng kiểm tra

```bash
cd quiz-app

# Lint code
npm run lint

# Build to verify no errors
npm run build
```

**Lưu ý**: Đây chủ yếu là kho lưu trữ chương trình học giáo dục. Không có kiểm tra tự động cho nội dung bài học. Việc xác thực được thực hiện thông qua:
- Hoàn thành bài tập bài học
- Chạy các ô notebook thành công
- Kiểm tra đầu ra so với kết quả mong đợi trong các giải pháp

## Hướng dẫn phong cách mã

### Mã Python
- Tuân theo hướng dẫn phong cách PEP 8
- Sử dụng tên biến rõ ràng, mô tả
- Bao gồm các bình luận cho các thao tác phức tạp
- Jupyter notebooks nên có các ô markdown giải thích khái niệm

### JavaScript/Vue.js (Ứng dụng kiểm tra)
- Tuân theo hướng dẫn phong cách Vue.js
- Cấu hình ESLint trong `quiz-app/package.json`
- Chạy `npm run lint` để kiểm tra và tự động sửa lỗi

### Tài liệu
- Các tệp markdown nên rõ ràng và có cấu trúc tốt
- Bao gồm các ví dụ mã trong các khối mã được bao quanh
- Sử dụng liên kết tương đối cho các tham chiếu nội bộ
- Tuân theo các quy ước định dạng hiện có

## Xây dựng và triển khai

### Triển khai ứng dụng kiểm tra

Ứng dụng kiểm tra có thể được triển khai lên Azure Static Web Apps:

1. **Yêu cầu trước**:
   - Tài khoản Azure
   - Kho lưu trữ GitHub (đã được fork)

2. **Triển khai lên Azure**:
   - Tạo tài nguyên Azure Static Web App
   - Kết nối với kho lưu trữ GitHub
   - Đặt vị trí ứng dụng: `/quiz-app`
   - Đặt vị trí đầu ra: `dist`
   - Azure tự động tạo workflow GitHub Actions

3. **Workflow GitHub Actions**:
   - Tệp workflow được tạo tại `.github/workflows/azure-static-web-apps-*.yml`
   - Tự động xây dựng và triển khai khi đẩy lên nhánh chính

### Tài liệu PDF

Tạo PDF từ tài liệu:

```bash
npm install
npm run convert
```

## Quy trình dịch thuật

**Quan trọng**: Dịch thuật được tự động hóa thông qua GitHub Actions sử dụng Co-op Translator.

- Dịch thuật được tự động tạo khi có thay đổi được đẩy lên nhánh `main`
- **KHÔNG tự dịch nội dung** - hệ thống sẽ xử lý việc này
- Workflow được định nghĩa trong `.github/workflows/co-op-translator.yml`
- Sử dụng dịch vụ Azure AI/OpenAI để dịch
- Hỗ trợ hơn 40 ngôn ngữ

## Hướng dẫn đóng góp

### Đối với người đóng góp nội dung

1. **Fork kho lưu trữ** và tạo nhánh tính năng
2. **Thực hiện thay đổi nội dung bài học** nếu thêm/cập nhật bài học
3. **Không sửa đổi các tệp đã dịch** - chúng được tự động tạo
4. **Kiểm tra mã của bạn** - đảm bảo tất cả các ô notebook chạy thành công
5. **Xác minh liên kết và hình ảnh** hoạt động chính xác
6. **Gửi yêu cầu kéo** với mô tả rõ ràng

### Hướng dẫn yêu cầu kéo

- **Định dạng tiêu đề**: `[Phần] Mô tả ngắn gọn về thay đổi`
  - Ví dụ: `[Regression] Sửa lỗi chính tả trong bài học 5`
  - Ví dụ: `[Quiz-App] Cập nhật các phụ thuộc`
- **Trước khi gửi**:
  - Đảm bảo tất cả các ô notebook chạy mà không có lỗi
  - Chạy `npm run lint` nếu sửa đổi ứng dụng kiểm tra
  - Xác minh định dạng markdown
  - Kiểm tra bất kỳ ví dụ mã mới nào
- **PR phải bao gồm**:
  - Mô tả về thay đổi
  - Lý do cho thay đổi
  - Ảnh chụp màn hình nếu có thay đổi giao diện người dùng
- **Quy tắc ứng xử**: Tuân theo [Quy tắc ứng xử mã nguồn mở của Microsoft](CODE_OF_CONDUCT.md)
- **CLA**: Bạn sẽ cần ký Thỏa thuận cấp phép cho người đóng góp

## Cấu trúc bài học

Mỗi bài học tuân theo một mẫu nhất quán:

1. **Kiểm tra trước bài giảng** - Kiểm tra kiến thức cơ bản
2. **Nội dung bài học** - Hướng dẫn và giải thích bằng văn bản
3. **Minh họa mã** - Ví dụ thực hành trong các notebook
4. **Kiểm tra kiến thức** - Xác minh sự hiểu biết trong suốt bài học
5. **Thử thách** - Áp dụng khái niệm một cách độc lập
6. **Bài tập** - Thực hành mở rộng
7. **Kiểm tra sau bài giảng** - Đánh giá kết quả học tập

## Tham khảo lệnh thông dụng

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

## Tài nguyên bổ sung

- **Bộ sưu tập Microsoft Learn**: [Các module Học máy cho người mới bắt đầu](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)
- **Ứng dụng kiểm tra**: [Kiểm tra trực tuyến](https://ff-quizzes.netlify.app/en/ml/)
- **Diễn đàn thảo luận**: [GitHub Discussions](https://github.com/microsoft/ML-For-Beginners/discussions)
- **Video hướng dẫn**: [Danh sách phát YouTube](https://aka.ms/ml-beginners-videos)

## Công nghệ chính

- **Python**: Ngôn ngữ chính cho các bài học học máy (Scikit-learn, Pandas, NumPy, Matplotlib)
- **R**: Triển khai thay thế sử dụng tidyverse, tidymodels, caret
- **Jupyter**: Notebook tương tác cho các bài học Python
- **R Markdown**: Tài liệu cho các bài học R
- **Vue.js 3**: Framework ứng dụng kiểm tra
- **Flask**: Framework ứng dụng web để triển khai mô hình học máy
- **Docsify**: Trình tạo trang tài liệu
- **GitHub Actions**: CI/CD và dịch thuật tự động

## Cân nhắc về bảo mật

- **Không có thông tin bí mật trong mã**: Không bao giờ cam kết API keys hoặc thông tin đăng nhập
- **Phụ thuộc**: Cập nhật các gói npm và pip thường xuyên
- **Đầu vào người dùng**: Ví dụ ứng dụng web Flask bao gồm xác thực đầu vào cơ bản
- **Dữ liệu nhạy cảm**: Các tập dữ liệu ví dụ là công khai và không nhạy cảm

## Xử lý sự cố

### Jupyter Notebooks

- **Vấn đề kernel**: Khởi động lại kernel nếu các ô bị treo: Kernel → Restart
- **Lỗi nhập khẩu**: Đảm bảo tất cả các gói cần thiết đã được cài đặt bằng pip
- **Vấn đề đường dẫn**: Chạy các notebook từ thư mục chứa chúng

### Ứng dụng kiểm tra

- **npm install thất bại**: Xóa bộ nhớ cache npm: `npm cache clean --force`
- **Xung đột cổng**: Thay đổi cổng với: `npm run serve -- --port 8081`
- **Lỗi xây dựng**: Xóa `node_modules` và cài đặt lại: `rm -rf node_modules && npm install`

### Bài học R

- **Không tìm thấy gói**: Cài đặt với: `install.packages("package-name")`
- **Kết xuất RMarkdown**: Đảm bảo gói rmarkdown đã được cài đặt
- **Vấn đề kernel**: Có thể cần cài đặt IRkernel cho Jupyter

## Ghi chú cụ thể về dự án

- Đây chủ yếu là một **chương trình học tập**, không phải mã sản xuất
- Tập trung vào **hiểu các khái niệm học máy** thông qua thực hành
- Các ví dụ mã ưu tiên **sự rõ ràng hơn tối ưu hóa**
- Hầu hết các bài học **tự chứa** và có thể hoàn thành độc lập
- **Giải pháp được cung cấp** nhưng người học nên thử bài tập trước
- Kho lưu trữ sử dụng **Docsify** để tài liệu web mà không cần bước xây dựng
- **Sketchnotes** cung cấp tóm tắt trực quan về các khái niệm
- **Hỗ trợ đa ngôn ngữ** giúp nội dung tiếp cận toàn cầu

---

**Tuyên bố miễn trừ trách nhiệm**:  
Tài liệu này đã được dịch bằng dịch vụ dịch thuật AI [Co-op Translator](https://github.com/Azure/co-op-translator). Mặc dù chúng tôi cố gắng đảm bảo độ chính xác, xin lưu ý rằng các bản dịch tự động có thể chứa lỗi hoặc không chính xác. Tài liệu gốc bằng ngôn ngữ bản địa nên được coi là nguồn tham khảo chính thức. Đối với các thông tin quan trọng, nên sử dụng dịch vụ dịch thuật chuyên nghiệp từ con người. Chúng tôi không chịu trách nhiệm cho bất kỳ sự hiểu lầm hoặc diễn giải sai nào phát sinh từ việc sử dụng bản dịch này.