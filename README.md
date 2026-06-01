# ElevenLabs Text-to-Speech (TTS) với Bộ tăng cường âm thanh DSP

Dự án này cung cấp một script Python để thực hiện chuyển đổi văn bản thành giọng nói (TTS) sử dụng mô hình **Eleven v3** mới nhất từ ElevenLabs với giọng đọc **Adam**. Script được thiết lập các tham số tối ưu cho chất giọng **Quyết đoán & Vững chãi (Dominant & Firm)**, đồng thời tích hợp một bộ lọc xử lý tín hiệu số (DSP) nội bộ bằng NumPy và SciPy để tăng cường chất lượng âm thanh chuyên nghiệp.

## 🚀 Tính năng nổi bật

- **Mô hình Eleven v3:** Sử dụng mô hình mới nhất hỗ trợ sắc thái biểu cảm tốt hơn.
- **Giọng đọc Adam:** Tự động gọi giọng Adam bản quyền của ElevenLabs (`pNInz6obpgDQGcFmaJgB`).
- **Tông giọng Quyết đoán, Vững chãi (Dominant & Firm):** Được tối ưu qua cấu hình ElevenLabs Voice Settings (`stability=0.75`, `similarity_boost=0.85`, bật `use_speaker_boost`).
- **Quy trình tăng cường DSP cục bộ:**
  - **Cắt ù nền (High-Pass Filter):** Loại bỏ tần số dưới 80Hz chống ồn và ù.
  - **Tăng ấm (Warmth EQ):** Boost dải âm trầm giọng nói tại 150Hz (+3.5 dB) để tăng nội lực, độ dày.
  - **Tăng rõ nét (Presence EQ):** Boost dải âm trung cao tại 3.2kHz (+2.5 dB) giúp phát âm rõ ràng, sắc nét.
  - **Bộ nén Dynamic Compressor:** Giúp cân bằng âm lượng tự động và tăng độ chắc chắn, đầy đặn của giọng đọc.
  - **Chuẩn hóa âm lượng (Peak Normalization):** Đưa đỉnh âm lượng của tệp tin về mức chuẩn -1 dB để âm thanh to, rõ ràng mà không bao giờ bị rè.
- **Tự động chuyển đổi dự phòng (Fallback):** Tự động thử nghiệm các định dạng chất lượng PCM từ cao xuống thấp (`pcm_44100` -> `pcm_24000` -> `pcm_22050` -> `pcm_16000`) tùy thuộc vào phân quyền gói tài khoản ElevenLabs của bạn.

---

## 🛠️ Hướng dẫn cài đặt

### 1. Cài đặt các thư viện phụ thuộc
Chạy lệnh sau để cài đặt các thư viện cần thiết:
```bash
python -m pip install elevenlabs numpy scipy python-dotenv
```

### 2. Cấu hình ElevenLabs API Key
1. Copy tệp `.env.example` thành tệp `.env`:
   ```bash
   copy .env.example .env
   ```
2. Mở tệp `.env` vừa tạo và thay thế `your_api_key_here` bằng ElevenLabs API Key của bạn:
   ```env
   ELEVENLABS_API_KEY=your_actual_api_key_here
   ```

---

## 📖 Hướng dẫn sử dụng

### 1. Chuyển đổi văn bản trực tiếp từ dòng lệnh (CLI)
Chạy script và truyền trực tiếp văn bản cần đọc:
```bash
python tts_adam_enhance.py -t "Xin chào bạn, đây là giọng đọc thử nghiệm của mô hình Eleven v3 được tăng cường âm thanh."
```
*Tệp tin kết quả mặc định sẽ được lưu tại `output_enhanced.wav` (đã tăng cường) và `output_raw.wav` (chất lượng gốc để so sánh).*

### 2. Chuyển đổi văn bản từ một tệp tin văn bản (txt)
Nếu bạn có đoạn văn bản dài, hãy lưu nó vào tệp `.txt` (ví dụ `document.txt`) và chạy:
```bash
python tts_adam_enhance.py -f document.txt -o output_documents.wav
```

### 3. Các tùy chọn tham số dòng lệnh khác

Bạn có thể thay đổi các cấu hình âm thanh mặc định bằng cách truyền thêm tham số:

| Tham số | Ý nghĩa | Mặc định |
| :--- | :--- | :--- |
| `-o`, `--output` | Đường dẫn lưu tệp WAV đã được tăng cường | `output_enhanced.wav` |
| `-r`, `--output-raw` | Đường dẫn lưu tệp WAV gốc chưa tăng cường | `output_raw.wav` |
| `--skip-enhance` | Bỏ qua bộ lọc DSP, chỉ tải file gốc từ API | Không bật |
| `--api-key` | Truyền trực tiếp API Key (nếu không dùng `.env`) | Không |
| `--stability` | Độ ổn định giọng nói (0.0 đến 1.0) | `0.75` |
| `--similarity` | Độ giống giọng nói gốc (0.0 đến 1.0) | `0.85` |
| `--style` | Mức độ cường điệu hóa phong cách giọng đọc (0.0 đến 1.0) | `0.0` |
| `--disable-speaker-boost` | Tắt chức năng tăng cường độ rõ của giọng nói gốc | Bật mặc định |
| `--warmth-gain` | Độ lợi tăng ấm dải trầm tại 150 Hz (dB) | `3.5` |
| `--clarity-gain` | Độ lợi tăng nét dải trung cao tại 3.2 kHz (dB) | `2.5` |
| `--threshold` | Ngưỡng kích hoạt bộ nén compressor (dB) | `-18.0` |
| `--ratio` | Tỉ lệ nén âm lượng của compressor | `3.0` |

**Ví dụ cấu hình tùy chỉnh nâng cao:**
```bash
python tts_adam_enhance.py -t "Đây là giọng đọc cấu hình tùy chỉnh." --warmth-gain 4.0 --clarity-gain 3.0 --threshold -15.0 --ratio 4.0
```

---

## 🌐 Giao diện Web GUI (Khuyên dùng)

Để giúp việc sử dụng thuận tiện và trực quan hơn, dự án được tích hợp sẵn một giao diện Web GUI cục bộ.

### 1. Khởi động Web Server
Chạy lệnh sau tại thư mục dự án:
```bash
python app.py
```
Màn hình console sẽ hiển thị thông tin máy chủ cục bộ đang chạy tại địa chỉ: `http://127.0.0.1:5000`.

### 2. Sử dụng trên Trình duyệt
- Truy cập vào đường link [http://127.0.0.1:5000](http://127.0.0.1:5000) trên trình duyệt.
- Bạn sẽ thấy giao diện tối (Dark Mode / Glassmorphism) hiện đại:
  - **Bảng điều khiển bên trái:** Cho phép bạn nhập văn bản (tối đa 5000 ký tự) và điều chỉnh các thông số sinh giọng nói của ElevenLabs (Stability, Similarity, Style, Speaker Boost).
  - **Bảng điều khiển bên phải:** Giúp bạn bật/tắt bộ lọc DSP cục bộ và tùy chỉnh linh hoạt các thông số nén âm lượng (Compressor) và độ lợi EQ trầm ấm (Warmth), sắc nét (Clarity).
  - **Nút "TẠO GIỌNG NÓI":** Thực hiện chuyển đổi và áp dụng bộ lọc tăng cường DSP trong vài giây.
  - **Trình phát âm thanh so sánh:** Nghe thử trực tuyến và tải về (WAV) cả hai phiên bản **Giọng đọc đã tăng cường (DSP Enhanced)** và **Giọng gốc (Raw)** để kiểm định chất lượng âm thanh.

