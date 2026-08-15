# Báo cáo LAB 16 — Cloud AI Environment Setup (GCP)

**Học viên:** Trương Công Đạt — 2A202601449
**Ngày:** 15/08/2026
**Project GCP:** `mystic-gradient-505507-g3` · Region `us-central1` · Node `e2-medium` (2 vCPU / 4 GB)

---

## Báo cáo ngắn

Hạ tầng được triển khai hoàn toàn bằng Terraform (16 tài nguyên, ~4 phút): VPC riêng, Cloud NAT
cho chiều ra, và một Compute Node **không có IP public** — truy cập SSH qua Identity-Aware Proxy.
Mô hình LightGBM huấn luyện trên 284.807 giao dịch chỉ mất **24,8 giây** với 2 vCPU, cho thấy bài
toán dữ liệu dạng bảng **không cần GPU**: một GPU T4 đắt gấp ~10 lần mà không nhanh hơn, vì
gradient boosting không tận dụng được tính song song ồ ạt của GPU.
Chất lượng mô hình đạt **AUC-ROC 0,9890** và **F1 0,8619** (Precision 0,9398 / Recall 0,7959):
bắt được 78/98 giao dịch gian lận với chỉ 5 báo động giả trên 56.864 giao dịch sạch. Ngược lại,
**Accuracy 0,9996 gần như vô nghĩa** vì một mô hình luôn đoán "không gian lận" cũng đạt 0,9983 —
đây là bẫy kinh điển của dữ liệu mất cân bằng (tỉ lệ gian lận chỉ 0,172%).
Về hiệu năng phục vụ, **latency 1,98 ms/dòng** nhưng **throughput đạt 49.582 dòng/giây** khi xử lý
theo lô 1.000 dòng — tức xử lý theo lô **nhanh gấp ~98 lần**, do chi phí cố định mỗi lần gọi
`predict()` được chia đều thay vì áp đảo như khi dự đoán từng dòng.
Chi phí thực đo khoảng **$0,09/giờ**, trong đó **Cloud NAT chiếm ~49%** — đắt hơn cả tiền máy ảo,
một điểm dễ bị bỏ sót khi ước tính chi phí hạ tầng ML.

---

## Chi tiết kết quả (`benchmark_result.json`)

| Metric | Kết quả |
|---|---|
| Thời gian load data | 2,598 s |
| Thời gian training | 24,685 s |
| Best iteration | 310 / 500 |
| AUC-ROC | 0,989038 |
| Accuracy | 0,999561 |
| F1-Score | 0,861878 |
| Precision | 0,939759 |
| Recall | 0,795918 |
| Inference latency (1 row) | 1,979 ms (median, 200 lần đo) |
| Inference throughput (1000 rows) | 49.582 dòng/giây |

**Confusion matrix:** TP = 78 · FP = 5 · FN = 20 · TN = 56.859 (tổng 25 lỗi / 56.962 giao dịch)

---

## Quá trình tinh chỉnh — 4 lần chạy

Lần chạy đầu tiên cho `best_iteration = 1`, tức mô hình chỉ dùng **một cây quyết định**. Thay vì
chấp nhận, đã tiến hành thí nghiệm có kiểm soát để tìm nguyên nhân:

| | v1 | v2 | v3 | **v4 (cuối)** |
|---|---|---|---|---|
| `min_child_samples` | 20 | 20 | 100 | **100** |
| `stopping_rounds` | 50 | 50 | 50 | **200** |
| Best iteration | 1 | 1 | 2 | **310** |
| AUC-ROC | 0,9517 | 0,9517 | 0,9717 | **0,9890** |
| F1 | 0,7273 | 0,7273 | 0,7813 | **0,8619** |
| Báo động giả (FP) | 42 | 42 | 19 | **5** |
| Training time | 3,4 s | 3,1 s | 3,2 s | **24,8 s** |
| Throughput | 454.978/s | 418.893/s | 370.259/s | **49.582/s** |

**Nguyên nhân thực sự** (xác định bằng thí nghiệm in AUC theo từng mốc số cây, không dùng early
stopping):

1. `min_child_samples = 20` (mặc định) quá nhỏ so với dữ liệu mất cân bằng. Tập train chỉ có ~394
   ca gian lận, nên mô hình sinh ra các lá tí hon **học thuộc từng ca riêng lẻ** → overfit. Hệ quả
   đo được: AUC đạt đỉnh ngay ở cây thứ 1 rồi tụt xuống 0,7574 ở cây thứ 5. Nâng lên 100 buộc mỗi
   lá phải có tối thiểu 100 mẫu chống lưng.
2. `stopping_rounds = 50` quá ngắn. Đường AUC có một "vùng trũng" dài hơn 50 vòng trước khi leo lên
   đỉnh thật quanh cây 300, nên early stopping dừng ngay trong vùng trũng và không bao giờ thấy
   được đỉnh. Nâng lên 200 giải quyết triệt để.

Giả thuyết ban đầu cho rằng cột `Time` gây nhiễu **đã bị bác bỏ bằng thực nghiệm**: bỏ cột này chỉ
làm AUC giảm nhẹ (0,9517 → 0,9415) và vẫn đỉnh ở cây 1.

---

## Đánh đổi accuracy ↔ latency

So sánh v1 (1 cây) với v4 (310 cây) cho một quan sát đáng chú ý:

| | v1 | v4 | Thay đổi |
|---|---|---|---|
| Latency 1 dòng | 1,726 ms | 1,979 ms | chậm **1,15×** |
| Thời gian/dòng khi chạy lô | 0,0022 ms | 0,0202 ms | chậm **9,2×** |

Mô hình nặng gấp 310 lần nhưng **latency gần như không đổi**, trong khi **throughput giảm 9 lần**.
Lý do: khi dự đoán một dòng, chi phí cố định (chuyển pandas → numpy, gọi vào thư viện C++) áp đảo
phần tính toán thật; khi chạy theo lô, chi phí đó được chia đều nên phần tính toán mới lộ ra.

**Hệ quả thiết kế:** hệ thống phục vụ từng request lẻ có thể dùng mô hình lớn gần như miễn phí về
độ trễ; ngược lại pipeline batch hàng triệu dòng phải trả giá cho từng cây bổ sung.

---

## Sử dụng tài nguyên và tối ưu chi phí

| Tài nguyên | Cấp phát | Thực dùng | Tỉ lệ |
|---|---|---|---|
| RAM | 3,8 GiB | 525 MiB | 13,5% |
| Disk | 30 GB (pd-ssd) | 3,2 GB | 12% |
| CPU | 2 vCPU | ~190% khi training | — |
| Network RX | — | 242 MB (dataset + thư viện, qua Cloud NAT) | — |
| Network TX | — | 1,4 MB | — |

Máy bị cấp thừa rõ rệt. Đề xuất right-sizing:

- `e2-medium` → `e2-small` (2 GB RAM): tiết kiệm ~50% tiền máy, vẫn dư gấp 4 lần nhu cầu thực.
- Đĩa `pd-ssd` 30 GB → `pd-balanced` 20 GB: tiết kiệm ~60% tiền đĩa.
- **Cloud NAT là khoản đắt nhất** (~$0,044/giờ, chiếm ~49% tổng chi phí) dù chỉ dùng để tải
  dependency lúc khởi tạo. Với workload chạy dài, nên cân nhắc dựng image đã cài sẵn thư viện
  (custom image / Artifact Registry + Private Google Access) để bỏ hẳn NAT.

**Ước tính chi phí:** ~$0,09/giờ · tổng chi phí thực tế cho toàn bộ buổi lab dưới $0,30.

---

## Sự cố gặp phải và cách xử lý

| Sự cố | Nguyên nhân | Xử lý |
|---|---|---|
| `terraform init` lỗi `connection forcibly closed` | Tuyến IPv6 từ ISP tới `registry.terraform.io` chập chờn; Terraform (Go) ưu tiên IPv6 | Tải provider bằng `curl -4` (ép IPv4), đối chiếu SHA256, đặt vào **provider mirror** cục bộ |
| Startup script thất bại, `exit status 127` | File `.sh` lưu theo chuẩn Windows (CRLF); shebang trở thành `/bin/bash\r` nên kernel không tìm thấy interpreter | Chuyển toàn bộ file `.sh` sang LF; cài thư viện thủ công trên VM đang chạy (sửa file sẽ buộc Terraform tạo lại VM) |
| `gcloud compute scp` báo không tìm thấy đường dẫn | `pscp` trên Windows không hiểu `~` (đó là ký hiệu của shell, không phải đường dẫn thật) và không nhận `.` làm đích | Dùng đường dẫn tuyệt đối `/home/<user>/...` và ghi rõ tên file đích |

---

## Dọn dẹp

`terraform destroy` xóa toàn bộ 16 tài nguyên. Đã kiểm chứng độc lập bằng `gcloud`: không còn VM,
disk, external IP, forwarding rule, Cloud NAT hay VPC `ai-vpc`. Chỉ còn lại VPC `default` và
service account mặc định của Compute Engine — cả hai do Google tự tạo và **không phát sinh chi phí**.
