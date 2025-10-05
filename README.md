PROJECT UIT_Student's Feedback 👋

Dataset : Tải và thao tác tại: https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback
Đây là bộ dataset từ VNUHCM-UIT, với target là sentiment

⚡TIỀN XỬ LÍ DỮ LIỆU
- Ở column Sentence là bằng tiếng Việt, nếu áp TF-IDF vào ngay sẽ bị các lỗi như các từ cùng nghĩa nhưng bị tách ra. VD: Ko - Không, các từ ghép như Sinh Viên bị tách thành 2 từ riêng
- Ta cần một thư viện có thể xử lí tiếng Việt. Để đơn giản mình dùng thư viện underthesea và import word_tokenize, đây là một thư viện về NLP của tiếng Việt (có thư viện khác nhưng build từ Java, yêu cầu máy có môi trường chạy Java)
- Ta không cần impute cột Sentence vì qua bước word_tokenize đã xử lí điều đó. Và không cần impute cho cột Topic vì không có dữ liệu khuyết
- Cột Topic ta cần phải xử lí One-Hot Encoder vì nó là thuộc tính Ordinal, tránh để mô hình hiểu nhầm là Nominal, phân cấp 0-1-2-3

⚡FIT DỮ LIỆU VÀO MÔ HÌNH 
- Tiến hành dùng GridSearchCV để tìm bộ tham số tối ưu
- Tiến hành thử lại bộ tham số tối ưu đó với bộ Validation
- In ra Classification với bộ Test

⚡LƯU Ý 
- Nên giới hạn kích thước TF-IDF để tránh việc kích thước vector quá lớn, tốn bộ nhớ, thời gian,... 
