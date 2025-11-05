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

@@ -24,21 +23,30 @@
db = client["food-del"]
collection = db["foods"]

def get_food_list():
    foods = list(
        collection.find({}, {"_id": 0, "name": 1, "price": 1, "description": 1})
    )
    return foods














def ask_gemini(question: str, food_list: list) -> str:
    """
    Gửi câu hỏi + dữ liệu món ăn cho Gemini AI, trả về text đẹp, hỗ trợ bán online
    """
    food_lines = []
    for f in food_list:
        price = f.get('price', 0)
        points = int(price / 100000 * 10)

        line = f"- {f['name']}: {f.get('description', 'Không có mô tả')} - {price}đ - Điểm tích lũy: {points}"
        food_lines.append(line)

@@ -73,55 +81,55 @@
    if not question:
        return jsonify({"reply": "Xin lỗi, bạn chưa nhập câu hỏi."}), 400

    food_list = get_food_list()
    reply = ask_gemini(question, food_list)

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

    app.debug = True
    app.run()