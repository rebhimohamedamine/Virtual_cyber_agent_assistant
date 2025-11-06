from fastapi import FastAPI, Request
from sentence_transformers import SentenceTransformer, util

app = FastAPI()
model = SentenceTransformer('all-MiniLM-L6-v2')  # Load once

def evaluate_agent(expected_output, actual_output):
    embeddings = model.encode([expected_output, actual_output], convert_to_tensor=True)
    cosine_score = util.cos_sim(embeddings[0], embeddings[1]).item()
    
    if cosine_score > 0.85:
        rating = "✅ Excellent (accurate & likely used KB)"
    elif cosine_score > 0.7:
        rating = "⚠️ Good (reasonable match)"
    elif cosine_score > 0.5:
        rating = "⚠️ Medium (partial match)"
    else:
        rating = "❌ Poor (low semantic match)"

    return {
        "semantic_similarity": round(cosine_score, 2),
        "rating": rating
    }

@app.post("/evaluate")
async def evaluate(request: Request):
    try:
        data_input = await request.json()
        
        # Handle both array and single object formats
        if isinstance(data_input, list):
            if len(data_input) == 0:
                return {"error": "Expected at least one item in the array."}
            data = data_input[0]
        elif isinstance(data_input, dict):
            data = data_input
        else:
            return {"error": "Expected JSON object or array."}
        
        user_question = data.get("user_question", "")
        expected = data.get("expected_answer", "")
        actual = data.get("actual_answer", "")
        
        result = evaluate_agent(expected, actual)
        return result
        
    except Exception as e:
        return {"error": f"JSON parsing error: {str(e)}"}
