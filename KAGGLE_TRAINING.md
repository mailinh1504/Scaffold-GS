Hướng dẫn chạy training trên Kaggle

Tổng quan
- Mục tiêu: chạy mô hình Scaffold-GS (với mở rộng identity) trên môi trường Kaggle Notebook / Kernel có GPU.
- Lưu ý: Kaggle cung cấp môi trường ephemeral: file hệ thống tạm thời (working dir) bị xóa sau session. Dữ liệu lớn nên được upload lên `Kaggle Datasets` và gắn vào Notebook.

Chuẩn bị
1. Chuẩn bị repo GitHub (hoặc ZIP): bạn có thể `git clone` trực tiếp trong kernel nếu bật "Internet: On".
2. Chuẩn bị dữ liệu: upload ảnh / masks / prototypes thành một `Kaggle Dataset` hoặc nén và tải lên `Input` của Notebook.
3. Tạo file `requirements-kaggle.txt` (đã cung cấp) hoặc `environment.yml` nếu muốn dùng conda.

Mẫu workflow trong một Kaggle Notebook
1) Mở Notebook: chọn `Notebook` -> `New Notebook` -> chọn `GPU` (nVidia). Bật `Internet: On` nếu cần clone repo.

2) Clone repo (nếu ở GitHub):

```bash
# ví dụ trong cell bash của Notebook
git clone https://github.com/<YOUR_USER>/<REPO>.git
cd <REPO>
```

3) Cài phụ thuộc:

```bash
# Trong cell bash
pip install -r requirements-kaggle.txt
# Nếu cần torch phù hợp với CUDA trên Kaggle, thay bằng wheel chính thức của Pytorch:
# pip install --index-url https://download.pytorch.org/whl/cu118 torch torchvision
```

4) Tải dataset (nếu bạn đã tạo Kaggle Dataset):

```bash
# cài kaggle CLI và cấu hình (đặt kaggle.json trong /root/.kaggle)
pip install kaggle
kaggle datasets download -d <your-username>/<dataset-slug> -p /kaggle/working/dataset --unzip
```

5) Chạy training (ví dụ):

```bash
python train.py --model_path ./output/scene --data_root /kaggle/working/dataset/images \
  --iterations 20000 --id_dim 32 --lambda_id2d 3.0 --lambda_idreg 0.001 \
  --id_warmup_start 500 --id_warmup_iters 2000
```

6) Lưu kết quả
- Kết quả lưu vào `./output/scene` trong workspace của Notebook; hãy nén và đẩy lên `Kaggle Datasets` hoặc tải xuống local.

```bash
zip -r output_scene.zip ./output/scene
# Nếu đã cấu hình kaggle CLI:
kaggle datasets create -p ./output --title "ScaffoldGS-run-001" --force
```

Tips & lưu ý
- Kiểm tra phiên bản CUDA/torch yêu cầu trước khi cài `torch` wheel. Nếu không chắc, dùng `pip install torch torchvision` mặc định rồi test.
- Kaggle có giới hạn bộ nhớ/ổ cứng; nếu dataset lớn, đưa dữ liệu vào `Kaggle Datasets` và mount.
- Để chạy nhiều lần, lưu checkpoints ra `./output` và upload lại thành dataset.

Tự động hóa (tuỳ chọn)
- Bạn có thể tạo 1 Notebook cell bash để chạy toàn bộ pipeline: clone repo, cài dependencies, download dataset, chạy training.

