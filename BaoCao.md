# BÁO CÁO THỰC HÀNH LAB 21: MLOps CI/CD Pipeline

**Học viên:** Nguyễn Anh Đức
**Mssv:** 2A202600387
**Bài Lab:** Lab 21 - Xây dựng CI/CD Pipeline cho AI Systems

---

## 1. Bộ siêu tham số đã chọn và lý do (Kết quả Bước 1)

Sau khi tiến hành huấn luyện (train) mô hình phân loại chất lượng rượu (RandomForestClassifier) và sử dụng MLflow để theo dõi các metrics, tôi đã thử nghiệm nhiều bộ cấu hình siêu tham số (hyperparameters) khác nhau.

**Bộ tham số cuối cùng được chọn:**
*   `n_estimators`: 200
*   `max_depth`: 20

**Lý do chọn:**
*   Khi sử dụng cấu hình này, mô hình đạt được độ chính xác (Accuracy) là **0.7540** (75.4%) trên tập dữ liệu kiểm thử (eval.csv). 
*   Kết quả này đã vượt qua ngưỡng an toàn (gate) yêu cầu là `accuracy >= 0.70` được thiết lập trong bước "Eval" của CI/CD Pipeline. Các bộ tham số có `n_estimators` thấp hơn thường cho ra độ chính xác thấp (dưới 60%), không đủ điều kiện để tự động deploy.

---

## 2. Các khó khăn gặp phải và cách giải quyết

Trong quá trình thực hiện Lab, hệ thống đã gặp khá nhiều trục trặc từ môi trường local đến hạ tầng Cloud. Dưới đây là các vấn đề cốt lõi và phương án giải quyết:

### Khó khăn 1: Xung đột phiên bản Python và thư viện
*   **Vấn đề:** Ban đầu sử dụng Python phiên bản quá mới (3.13), dẫn đến lỗi không thể cài đặt được các thư viện cũ trong file `requirements.txt` (đặc biệt là numpy, scipy).
*   **Giải pháp:** Hạ cấp phiên bản Python xuống bản ổn định hơn (3.10/3.11), xóa môi trường ảo cũ và tạo lại một `.venv` hoàn toàn mới. Sau đó chạy `pip install -r requirements.txt` thành công.

### Khó khăn 2: Sai lệch hạ tầng (AWS thay vì GCP)
*   **Vấn đề:** Các file cấu hình mẫu mặc định của bài Lab (trong `serve.py`, `.github/workflows/mlops.yml` và `requirements.txt`) đều thiết lập cho Google Cloud Storage (GCP). Tuy nhiên, hạ tầng thực tế đang sử dụng lại là Amazon Web Services (AWS) với S3 và EC2. Hệ thống báo lỗi không tìm thấy Google Credentials.
*   **Giải pháp:** Tiến hành cấu trúc lại toàn bộ code:
    1.  Xóa `google-cloud-storage`, `dvc[gs]` và thay bằng `boto3`, `dvc[s3]` trong file requirements.
    2.  Sửa code trong `src/serve.py` để dùng `boto3.client('s3')` tải model.pkl.
    3.  Viết lại Github Actions Workflow (`mlops.yml`) để dùng `AWS_ACCESS_KEY_ID` thay vì service account JSON của Google.

### Khó khăn 3: Phân quyền IAM (AccessDenied khi dùng DVC và EC2)
*   **Vấn đề:** Lệnh `dvc push` thất bại do user IAM không có quyền truy cập S3. Ngoài ra, máy ảo EC2 cũng không thể tải model từ S3 do thiếu credentials.
*   **Giải pháp:** 
    *   Chạy lệnh AWS CLI để cấp policy `AmazonS3FullAccess` cho user IAM ở local.
    *   Tạo riêng một IAM Role (`mlops-ec2-role`) có quyền Read S3 và đính kèm (attach) vào EC2 Instance Profile để EC2 có thể tự động xác thực mà không cần hard-code key vào bên trong máy ảo.

### Khó khăn 4: Lỗi Deploy trên GitHub Actions
*   **Vấn đề:** Job `Deploy` liên tục thất bại do lỗi không đọc được SSH Key (`ssh: no key found`) và không thể kết nối vào port 8000 của API. Hơn nữa, server EC2 ban đầu là một máy trắng, chưa có python/fastapi.
*   **Giải pháp:**
    *   Sửa lỗi copy-paste thiếu dòng trong GitHub Secret `VM_SSH_KEY`. Cập nhật `VM_USER` thành `ec2-user`.
    *   Dùng SSH vào cấu hình máy EC2: Cài Python3, Boto3, Uvicorn, FastAPI. Viết file cấu hình systemd (`/etc/systemd/system/mlops-serve.service`) để biến Uvicorn thành background service có khả năng tự động khởi động lại.
    *   Mở inbound rule cổng 8000 trên Security Group của EC2 để API có thể nhận request. Kết quả là lệnh `curl /predict` đã hoạt động xuất sắc.

---
*(Báo cáo kết thúc)*
