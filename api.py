import torch
import uvicorn
import traceback
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModelForTokenClassification

# Biến toàn cục lưu mô hình
tokenizer = None
model = None
id2label = None


# 1. CƠ CHẾ LIFESPAN (Chuẩn mới thay thế cho on_event)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Khối lệnh này chạy TRƯỚC KHI server mở cửa đón khách
    global tokenizer, model, id2label
    print("Đang tải mô hình NER từ './ner-pii-vi'...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("./ner-pii-vi")
        model = AutoModelForTokenClassification.from_pretrained("./ner-pii-vi")
        id2label = model.config.id2label

        if torch.cuda.is_available():
            model.to("cuda")

        print("=> Tải mô hình thành công và sẵn sàng nhận yêu cầu!")
    except Exception as e:
        print(f"Lỗi khi tải mô hình: {e}")

    yield  # Giao quyền điều khiển lại cho FastAPI để chạy server

    # Khối lệnh này chạy SAU KHI bạn tắt server
    print("Đang đóng API. Tạm biệt!")


# 2. KHỞI TẠO FASTAPI (Gắn lifespan vào đây)
app = FastAPI(
    title="SECUREPREP NER API",
    description="API nhận diện thông tin cá nhân và nhạy cảm trong văn bản.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 3. KHAI BÁO CẤU TRÚC DỮ LIỆU ĐẦU VÀO/RA
class PredictRequest(BaseModel):
    text: str


class Entity(BaseModel):
    field_name: str
    text: str
    start: int
    end: int


class PredictResponse(BaseModel):
    total_entities: int
    entities: List[Entity]


# 4. HÀM XỬ LÝ LÕI
def predict_full_document(full_text: str) -> List[Dict[str, Any]]:
    """
    Predict NER trên toàn bộ văn bản bằng sliding window — nhất quán với lúc train.

    Lúc train (ner_train.py), encode_records() tokenize trực tiếp toàn bộ text với:
        tokenizer(text, truncation=True, max_length=MAX_LEN,
                  stride=STRIDE, return_overflowing_tokens=True, ...)
    → Hàm này làm đúng như vậy, KHÔNG cắt theo '\\n' trước rồi mới sliding window.

    Tham số phải khớp ner_train.py:
        MAX_LEN = 384, STRIDE = 128
    """
    model.eval()

    def decode_entities(offsets: List[List[int]], ids: List[int], base_offset: int) -> List[Dict[str, Any]]:
        """Decode BIO labels cho một window → danh sách span tuyệt đối trong full_text."""
        entities = []
        cur = None
        for (cs, ce), pid in zip(offsets, ids):
            if cs == ce:          # token đặc biệt ([CLS], [SEP], padding)
                continue

            lab = id2label.get(int(pid), "O")
            if lab == "O" or "-" not in lab:
                if cur:
                    entities.append(cur)
                    cur = None
                continue

            tag, fld = lab.split("-", 1)
            abs_s = cs + base_offset
            abs_e = ce + base_offset

            if cur and tag == "I" and cur["field_name"] == fld:
                cur["end"] = max(cur["end"], abs_e)
            else:
                if cur:
                    entities.append(cur)
                cur = {"field_name": fld, "start": abs_s, "end": abs_e}

        if cur:
            entities.append(cur)
        return entities

    all_entities = []
    with torch.no_grad():
        # -------------------------------------------------------------------
        # Áp dụng sliding window trực tiếp trên TOÀN BỘ VĂN BẢN,
        # giống hệt encode_records() trong ner_train.py:
        #   tokenizer(text, truncation=True, max_length=MAX_LEN,
        #             stride=STRIDE, return_overflowing_tokens=True, ...)
        # offset_mapping của fast tokenizer đã là vị trí ký tự tuyệt đối
        # trong full_text → base_offset = 0.
        # -------------------------------------------------------------------
        enc = tokenizer(
            full_text,
            return_offsets_mapping=True,
            return_overflowing_tokens=True,
            truncation=True,
            max_length=384,   # MAX_LEN — khớp với ner_train.py
            stride=128,       # STRIDE  — khớp với ner_train.py
        )

        offsets_batch = enc.pop("offset_mapping")
        enc.pop("overflow_to_sample_mapping", None)

        for i in range(len(offsets_batch)):
            window_inputs = {}
            for k, v in enc.items():
                row = v[i]
                if not isinstance(row, list):
                    row = list(row)
                # v[i] là list 1-D → wrap thêm batch dim → tensor (1, seq_len)
                window_inputs[k] = torch.tensor([row], dtype=torch.long, device=model.device)

            logits = model(**window_inputs).logits[0]
            ids = logits.argmax(-1).tolist()
            offsets = offsets_batch[i]
            all_entities.extend(decode_entities(offsets, ids, base_offset=0))

    # Gộp các span trùng / chồng lấp phát sinh từ vùng stride (overlap giữa các window)
    all_entities.sort(key=lambda x: (x["field_name"], x["start"], x["end"]))
    merged_entities = []
    for ent in all_entities:
        if not merged_entities:
            merged_entities.append(ent)
            continue

        prev = merged_entities[-1]
        if ent["field_name"] == prev["field_name"] and ent["start"] <= prev["end"]:
            # Chồng lấp hoặc liền kề → mở rộng span hiện tại
            prev["end"] = max(prev["end"], ent["end"])
        else:
            merged_entities.append(ent)

    for e in merged_entities:
        e["text"] = full_text[e["start"]:e["end"]]

    return merged_entities


# 5. ĐỊNH NGHĨA API ENDPOINT
@app.post("/predict", response_model=PredictResponse)
async def predict_entities(request: PredictRequest):
    if not model or not tokenizer:
        raise HTTPException(status_code=500, detail="Mô hình chưa được tải hoàn tất.")

    if not request.text.strip():
        return PredictResponse(total_entities=0, entities=[])

    try:
        entities = predict_full_document(request.text)
        return PredictResponse(total_entities=len(entities), entities=entities)
    except Exception as e:
        print("[predict] Internal error:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Lỗi: {str(e)}")


@app.get("/")
async def root():
    return {"message": "API Nhận diện PII/SPII đang hoạt động ổn định!"}


@app.get("/health")
async def health():
    """Kiểm tra trạng thái sức khỏe của API"""
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None
    }


# 6. GIÚP SERVER KHÔNG BỊ TẮT KHI BẤM NÚT RUN
if __name__ == "__main__":
    print("Đang khởi động Server uvicorn...")
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)