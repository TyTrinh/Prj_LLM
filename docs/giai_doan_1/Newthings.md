# Câu hỏi 1: Tại sao nếu giữ LR cố định thì "Nổ Gradient ở đầu" và "Không thể hội tụ sâu ở cuối"?
- Bản chất khi vừa khởi tạo
    + Giai đoạn đầu, các tham số được khởi tạo ngẫu nhiên từ phân phối Gaussian -> Mô hình mù tịt về cấu trúc ngữ pháp
    -> Khi đưa câu đầu tiên vào, mô hình dự đoán sai lệch cực lớn
    -> Hàm loss ném ra gradient khổng lồ
- Nổ gradient ở những bước đầu bởi vì:
    + Khi dùng ngay lr quá lớn -> bước nhảy lớn thì lúc ban đầu lại chưa rõ phương hướng sẽ đẩy các trọng số văng ra khỏi miền học ổn định của nó
    -> Ma trận Attention tràn số, sinh ra giá trị NaN
- Tác dụng của warmup 200 steps đầu:
    + 