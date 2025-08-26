from openai import OpenAI
import os
from dotenv import load_dotenv
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import streamlit as st


# Инициализация моделей
@st.cache_resource
def load_model():
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    return tokenizer, model

tokenizer, model = load_model()

# OpenRouter клиент
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

# UI
st.title("Анализ телефонного диалога")
st.write("Введите текст диалога, и получите советы по улучшению коммуникации.")

query = st.text_area("Диалог", placeholder="Введите текст диалога здесь...")

if st.button("Анализировать"):
    if query.strip():
        # 1. Классификация
        inputs = tokenizer(query, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
        predicted_class_id = logits.argmax().item()
        a = model.config.id2label[predicted_class_id]

        st.write(f"**Классификация:** {a}")

        # 2. Генерация советов через LLM
        with st.spinner("Генерирую советы..."):
            completion = client.chat.completions.create(
                model="qwen/qwen2.5-vl-32b-instruct:free",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a master of communication. "
                            "You will receive a phone dialog {query} as one text block "
                            "and its classification {a} as positive or negative. "
                            "Your goal is to give maximum two short advices how to make communication better."
                        ),
                    },
                    {"role": "user", "content": query},
                    {"role": "user", "content": f"Classification: {a}"},
                ],
            )

        response = completion.choices[0].message.content
        st.subheader("Советы:")
        st.write(response)
    else:
        st.warning("Введите текст диалога.")