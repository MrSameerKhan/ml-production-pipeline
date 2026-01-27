import os, json
import numpy as np
import joblib
import tensorflow as tf
from flask import Flask, request, Response

app = Flask(__name__)

MODEL_DIR = "/opt/ml/model"
SAVEDMODEL_PATH = os.path.join(MODEL_DIR, "saved_model")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "preprocess.joblib")
LABELMAP_PATH = os.path.join(MODEL_DIR, "label_map.json")

model = None
vectorizer = None
id_to_label = None
infer_fn = None

def load_assets():
    global model, vectorizer, id_to_label, infer_fn
    if model is None:
        model = tf.saved_model.load(SAVEDMODEL_PATH)
        infer_fn = model.signatures["serving_default"]
    if vectorizer is None:
        vectorizer = joblib.load(VECTORIZER_PATH)
    if id_to_label is None:
        with open(LABELMAP_PATH, "r", encoding="utf-8") as f:
            m = json.load(f)
        # keys in json are strings
        id_to_label = {int(k): v for k, v in m["id_to_label"].items()}

@app.get("/ping")
def ping():
    try:
        load_assets()
        return Response("OK", status=200, mimetype="text/plain")
    except Exception as e:
        return Response(str(e), status=500, mimetype="text/plain")

@app.post("/invocations")
def invocations():
    load_assets()
    payload = request.get_json(force=True)

    text = payload.get("text")
    if text is None:
        return Response("Missing 'text' in JSON body", status=400)

    texts = [text] if isinstance(text, str) else list(text)
    X = vectorizer.transform(texts).toarray().astype(np.float32)

    out = infer_fn(tf.constant(X))
    probs = list(out.values())[0].numpy()
    pred_ids = probs.argmax(axis=1)
    preds = [id_to_label[int(i)] for i in pred_ids]

    return Response(
        json.dumps({"predictions": preds, "probabilities": probs.tolist()}),
        status=200,
        mimetype="application/json",
    )

if __name__ == "__main__":
    # SageMaker expects the container to listen on 8080
    app.run(host="0.0.0.0", port=8080)
