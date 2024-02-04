# Image captioning (Python 3.7.2)

# Link pretrain ImageNet các model đã dùng
-	Resnet-101: https://download.pytorch.org/models/resnet101-5d3b4d8f.pth

# Cấu trúc thư mục
- ACV_Final: Thư mục tên project
  - ```coco```: Thư mục chứa bộ dữ liệu, split annotation
  -  ```log```: Thư mục chứa các file log khi train
  -  ```model```: Thư mục định nghĩa mô hình Resnet-101 và là nơi mặc định lưu model được train
  -  ```self_collect```: Thư mục chứa dữ liệu nhóm tự thu thập
  -  ```caption.py```: Đưa ra kết quả mô tả ảnh trên 1 ảnh bất kỳ
  -  ```config.py```: File chứa các tinh chỉnh (training, đường dẫn, evaluate). Chi tiết được chú thích bên trong
  -  ```data_loader.py```: Định nghĩa custom data loader để training trong Pytorch
  -  ```data_reader.py```: Đọc dữ liệu tập coco và dữ liệu tự thu thập, sinh file .hdf5
  -  ```evaluate.py```: Evaluate model trên (mặc định tập test)
  -  ```main.py```: Huấn luyện mô hình (mặc định có Attention + Teacher forcing)
  -  ```util.py```: Các file chi tiết về quá trình train, valid, save model


# Setup
  - B1: Download dữ liệu COCO train, valid, nhóm thu thập và Andrej Karpathy's splits tại đây:
	-	https://images.cocodataset.org/zips/train2014.zip
	-	http://images.cocodataset.org/zips/val2014.zip
	-	https://drive.google.com/file/d/17Ug4woS4sVGr4wltHt-cYlgoYD2rM-X7/view?usp=sharing
	-	http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip

  - B2: Giải nén các file đã down (trừ self_collect.zip) ở trên vào thư mục ```coco```. Giải nén self_collect.zip vào thư mục ```self_collect```
  - B3: Chạy file ```data_reader.py``` để sinh file .hdf5


# Tiến hành thực nghiệm
  -	B1: Chạy main.py để train model (mặc định LSTM có Attention + Teacher forcing).
		+	Sau mỗi Epoch, mô hình sẽ được lưu tại thư mục model (ở đây sẽ lưu cả mô hình tốt nhất và mô hình ứng với epoch hiện tại)
		+ 	Nếu train xong 1 mô hình mà muốn train từ đầu/train mô hình khác. Cần đưa file model có đuôi .tar trong thư mục model ra một nơi khác để tránh mất model
		+	Để train không có Attention thì vào file ```main.py```. Sửa "decoder = DecoderWithAttention" -> "decoder = DecoderNoAttention".
		+ 	Nếu muốn tiếp tục train mô hình, sửa biến "checkpoint" trong ```config.py``` sao cho khớp đường dẫn
		( Các thông tin về quá trình train được in ra màn hình. Đồ thị train, valid được plot thủ công qua đó )
	
  -	B2: Chạy evaluate.py để đánh giá mô hình trên tập test. Chú ý model_path phải khớp với model đã train ở bước trên

  -	B3: Chạy file caption.py để minh họa kết quả phân loại 1 ảnh bất kỳ. Chú ý img_path và model_path phải khớp
