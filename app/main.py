from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from transformers import BertTokenizerFast, BertForSequenceClassification
import torch

app = FastAPI()

# Load model and tokenizer
tokenizer = BertTokenizerFast.from_pretrained("./model")
model = BertForSequenceClassification.from_pretrained("./model")
model.eval()

class TextIn(BaseModel):
    text: str

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
        <head><title>Sentiment Predictor</title></head>
        <body>
            <h2>Enter your text:</h2>
            <form action="/predict_form" method="post">
                <input type="text" name="text" style="width:300px" />
                <input type="submit" value="Analyze Sentiment" />
            </form>
        </body>
    </html>
    """

@app.post("/predict_form", response_class=HTMLResponse)
async def predict_form(request: Request):
    form = await request.form()
    text = form["text"]
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = torch.argmax(logits).item()
        label_map = {0: "negative", 1: "neutral", 2: "positive"}
        sentiment = label_map[predicted_class_id]
    return f"<h2>Sentiment: <span style='color:blue'>{sentiment}</span></h2><br><a href='/'>Try another</a>"
