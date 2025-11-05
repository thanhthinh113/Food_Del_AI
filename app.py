from flask import Flask, jsonify, request
from flask_cors import CORS
from pymongo import MongoClient
from bson.objectid import ObjectId
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer  
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai  
import os
from dotenv import load_dotenv

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

load_dotenv(".env")  
api_key = os.getenv("GOOGLE_API_KEY")  
MONGO_URI = os.getenv("MONGO_URI")
MODEL_NAME = os.getenv("MODEL_NAME")
genai.configure(api_key=api_key)
model = genai.GenerativeModel(MODEL_NAME)

client = MongoClient(MONGO_URI)
db = client["food-del"]
collection = db["foods"]

# 💡 KHỞI TẠO CACHE GLOBAL CHO DANH SÁCH MÓN ĂN
CACHED_FOOD_LIST = [] 

def load_initial_data():
    """Tải danh sách món ăn vào cache khi ứng dụng khởi động."""
    global CACHED_FOOD_LIST
    try:
        foods = list(
            collection.find({}, {"_id": 0, "name": 1, "price": 1, "description": 1})
        )
        CACHED_FOOD_LIST = foods
        print(f"✅ Đã tải thành công {len(CACHED_FOOD_LIST)} món ăn vào cache.")
    except Exception as e:
        print(f"❌ Lỗi khi tải dữ liệu từ MongoDB: {e}")
        
def ask_gemini(question: str, food_list: list) -> str:
    """
    Gửi câu hỏi + dữ liệu món ăn cho Gemini AI, trả về text đẹp, hỗ trợ bán online
    """
    food_lines = []
    for f in food_list:
        price = f.get('price', 0)
        # Đảm bảo tính toán điểm không gây lỗi
        points = int(price / 100000 * 10) if price and price >= 100000 else 0 
        line = f"- {f['name']}: {f.get('description', 'Không có mô tả')} - {price}đ - Điểm tích lũy: {points}"
        food_lines.append(line)
    
    food_text = "\n".join(food_lines)
    
    prompt = f"""
Bạn là chatbot nhà hàng Tomato, chỉ bán online, địa chỉ website: https://www.tomato.com
Hỗ trợ thanh toán qua Stripe, phí giao hàng: 30.000đ.
Khách mua món sẽ được tích điểm: 100.000đ = 10 điểm.

Danh sách món ăn hiện có:
{food_text}

Hướng dẫn AI:
- Chỉ trả lời đúng ý câu hỏi, không lan man.
- Nếu câu hỏi liên quan món ăn, điểm tích lũy, giá hoặc loại món, mới thêm thông tin danh sách hoặc điểm.
- Nếu câu hỏi về đặt món, thanh toán, giờ mở cửa… chỉ hướng dẫn cách đặt và thanh toán online.
- Không liệt kê tất cả món ăn hoặc thông tin không liên quan.
- Nhấn mạnh phí ship, cách thanh toán online và tích điểm nếu phù hợp.
- Trả lời ngắn gọn, tự nhiên, dễ đọc, không dùng dấu ** hay *.

Khách hàng hỏi: {question}
"""
    response = model.generate_content(prompt)
    return response.text.strip()


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    question = data.get("message", "").strip()
    if not question:
        return jsonify({"reply": "Xin lỗi, bạn chưa nhập câu hỏi."}), 400

    # 💡 SỬ DỤNG DỮ LIỆU ĐÃ CACHE, không cần truy vấn DB lại
    reply = ask_gemini(question, CACHED_FOOD_LIST)

    reply_clean = reply.replace("**", "").replace("*", "")
    return jsonify({"reply": reply_clean})


def top_recommend(food_index, cosine_similarity, top_n=4):
    score_similarity = list(enumerate(cosine_similarity[food_index]))
    sort = sorted(score_similarity, key=lambda x: x[1], reverse=True)[1 : top_n + 1]
    recommendation_index = [score[0] for score in sort]
    return recommendation_index


@app.route("/recommend/<food_id>", methods=["GET"])
def recommendation(food_id):
    object_id = ObjectId(food_id)

    # Lấy dữ liệu
    food = collection.find({}, {"description": 1, "image": 1, "price": 1, "name": 1})
    food_list = list(food)
    df = pd.DataFrame(food_list)

    # Ghép text để TF-IDF
    df["combined"] = df["description"] + " " + df["price"].astype(str)
    df["price"] = df["price"].astype(float)

    # TF-IDF + cosine
    vectorize = TfidfVectorizer()
    tfidf_matrix = vectorize.fit_transform(df["combined"])
    cosine_similarities = cosine_similarity(tfidf_matrix, tfidf_matrix)

    food_idx = df.index[df["_id"] == object_id][0]
    top_index = top_recommend(food_idx, cosine_similarities)

    result = []
    for idx in top_index:
        result.append(
            {
                "name": df.iloc[idx]["name"],
                "description": df.iloc[idx]["description"],
                "price": df.iloc[idx]["price"],
                "id": str(df.iloc[idx]["_id"]),
                "image": df.iloc[idx]["image"],
            }
        )
    return jsonify(result)

if __name__ == "__main__":
    load_initial_data() # 💡 Tải dữ liệu vào cache khi chạy
    app.debug = True
    app.run()