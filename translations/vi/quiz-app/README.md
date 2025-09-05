<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "6d130dffca5db70d7e615f926cb1ad4c",
  "translation_date": "2025-09-05T19:47:41+00:00",
  "source_file": "quiz-app/README.md",
  "language_code": "vi"
}
-->
# Câu hỏi trắc nghiệm

Các câu hỏi trắc nghiệm này là bài kiểm tra trước và sau bài giảng trong chương trình học ML tại https://aka.ms/ml-beginners

## Cài đặt dự án

```
npm install
```

### Biên dịch và tải lại nhanh cho phát triển

```
npm run serve
```

### Biên dịch và tối ưu hóa cho sản xuất

```
npm run build
```

### Kiểm tra và sửa lỗi tệp

```
npm run lint
```

### Tùy chỉnh cấu hình

Xem [Tham khảo cấu hình](https://cli.vuejs.org/config/).

Công nhận: Cảm ơn phiên bản gốc của ứng dụng câu hỏi này: https://github.com/arpan45/simple-quiz-vue

## Triển khai lên Azure

Dưới đây là hướng dẫn từng bước để giúp bạn bắt đầu:

1. Fork một kho lưu trữ GitHub  
Đảm bảo mã ứng dụng web tĩnh của bạn nằm trong kho lưu trữ GitHub. Fork kho lưu trữ này.

2. Tạo một Azure Static Web App  
- Tạo tài khoản [Azure](http://azure.microsoft.com)  
- Truy cập [Azure portal](https://portal.azure.com)  
- Nhấp vào “Create a resource” và tìm kiếm “Static Web App”.  
- Nhấp vào “Create”.  

3. Cấu hình Static Web App  
- Cơ bản:  
  - Subscription: Chọn gói đăng ký Azure của bạn.  
  - Resource Group: Tạo một nhóm tài nguyên mới hoặc sử dụng nhóm hiện có.  
  - Name: Đặt tên cho ứng dụng web tĩnh của bạn.  
  - Region: Chọn khu vực gần người dùng của bạn nhất.  

- #### Chi tiết triển khai:  
  - Source: Chọn “GitHub”.  
  - GitHub Account: Cấp quyền cho Azure truy cập tài khoản GitHub của bạn.  
  - Organization: Chọn tổ chức GitHub của bạn.  
  - Repository: Chọn kho lưu trữ chứa ứng dụng web tĩnh của bạn.  
  - Branch: Chọn nhánh bạn muốn triển khai từ.  

- #### Chi tiết xây dựng:  
  - Build Presets: Chọn framework mà ứng dụng của bạn được xây dựng (ví dụ: React, Angular, Vue, v.v.).  
  - App Location: Chỉ định thư mục chứa mã ứng dụng của bạn (ví dụ: / nếu nằm ở thư mục gốc).  
  - API Location: Nếu bạn có API, chỉ định vị trí của nó (tùy chọn).  
  - Output Location: Chỉ định thư mục nơi đầu ra xây dựng được tạo (ví dụ: build hoặc dist).  

4. Xem lại và tạo  
Xem lại cài đặt của bạn và nhấp vào “Create”. Azure sẽ thiết lập các tài nguyên cần thiết và tạo một workflow GitHub Actions trong kho lưu trữ của bạn.

5. Workflow GitHub Actions  
Azure sẽ tự động tạo một tệp workflow GitHub Actions trong kho lưu trữ của bạn (.github/workflows/azure-static-web-apps-<name>.yml). Workflow này sẽ xử lý quá trình xây dựng và triển khai.

6. Theo dõi triển khai  
Truy cập tab “Actions” trong kho lưu trữ GitHub của bạn.  
Bạn sẽ thấy một workflow đang chạy. Workflow này sẽ xây dựng và triển khai ứng dụng web tĩnh của bạn lên Azure.  
Khi workflow hoàn tất, ứng dụng của bạn sẽ hoạt động trên URL Azure được cung cấp.

### Tệp Workflow ví dụ

Dưới đây là ví dụ về tệp workflow GitHub Actions:  
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

### Tài nguyên bổ sung  
- [Tài liệu Azure Static Web Apps](https://learn.microsoft.com/azure/static-web-apps/getting-started)  
- [Tài liệu GitHub Actions](https://docs.github.com/actions/use-cases-and-examples/deploying/deploying-to-azure-static-web-app)  

---

**Tuyên bố miễn trừ trách nhiệm**:  
Tài liệu này đã được dịch bằng dịch vụ dịch thuật AI [Co-op Translator](https://github.com/Azure/co-op-translator). Mặc dù chúng tôi cố gắng đảm bảo độ chính xác, xin lưu ý rằng các bản dịch tự động có thể chứa lỗi hoặc không chính xác. Tài liệu gốc bằng ngôn ngữ bản địa nên được coi là nguồn thông tin chính thức. Đối với các thông tin quan trọng, khuyến nghị sử dụng dịch vụ dịch thuật chuyên nghiệp bởi con người. Chúng tôi không chịu trách nhiệm cho bất kỳ sự hiểu lầm hoặc diễn giải sai nào phát sinh từ việc sử dụng bản dịch này.