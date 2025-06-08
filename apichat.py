from flask import Flask, request, jsonify
from pyngrok import ngrok
import threading
import uuid
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline
from peft import PeftModel
from peft import AutoPeftModelForCausalLM
import torch
from flask_cors import CORS
app = Flask(__name__)
CORS(app)
# Giả lập LangChain messages và graph
from langchain_core.messages import AIMessage


from main import build_rag, InputQA, OutputQA

# Tạo Flask app

import re
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from typing import Literal, OrderedDict,     Union
from langchain_huggingface import ChatHuggingFace
from langchain_core.tools import tool
from typing_extensions import TypedDict
from typing import Annotated, Literal , List , Optional , Any
from pydantic import BaseModel, Field , field_validator, ValidationInfo , model_validator
from langgraph.graph.message import AnyMessage , add_messages
from langgraph.graph import StateGraph, MessagesState, START, END
import sys
import os
from langgraph.types import Command, interrupt
from flask import Flask, request, jsonify
import requests
from datetime import datetime
from langgraph.checkpoint.memory import MemorySaver
from langgraph.managed import IsLastStep, RemainingSteps
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage , SystemMessage, RemoveMessage
from langchain_core.runnables import Runnable
import uuid
from langgraph.prebuilt import create_react_agent


# model_id = "meta-llama/Llama-3.2-1B-Instruct"
model_id = "yuh0512/Llama-3.2-1B-Instruct-MEdQuAD-v7"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id,device_map="auto",
    torch_dtype=torch.float16,)

max_token = 128

model_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=max_token,
    pad_token_id=tokenizer.eos_token_id,
    # device="cuda" nếu muốn thêm thiết bị cụ thể
)

# Các tham số kiểm soát quá trình sinh văn bản
gen_kwargs = {
    "temperature": 0.1,  # độ sáng tạo
    "top_k": 10,         # chọn từ tốt nhất trong top 50
    "top_p": 0.7       # nucleus sampling (chọn từ trong xác suất tích lũy 95%)
}

llm = HuggingFacePipeline(
    pipeline=model_pipeline,
    model_kwargs=gen_kwargs
)
chat = ChatHuggingFace(llm = llm ,tokenizer=tokenizer)

from langgraph.checkpoint.memory import InMemorySaver

# Tạo đối tượng lưu trữ bộ nhớ
memory = InMemorySaver()

system_prompt = """
You are an intelligent AI medical assistant integrated into the MedicalCare system. Your role is to assist users by providing accurate health information and offering basic medical advice.

Respond to user questions with clear, accurate, and helpful information to support their healthcare needs.
"""
graph = create_react_agent(
    chat,
    tools=[],
    prompt=system_prompt,
    # checkpointer = memory,
)

config = {
    "configurable": {
    "thread_id": str(uuid.uuid4()),
    "user_id": "huy"
    }
}

@app.route('/chatbot', methods=['POST'])
def chatbot_v2():
    data = request.get_json()
    user_input = data.get("user_input", "")
    print(user_input)


    human_command = Command(update={"messages": user_input})
    final_response = ""

    for output in graph.stream(human_command, config=config, stream_mode="values"):
        messages = output.get("messages", [])
        for msg in messages:
            if isinstance(msg, AIMessage):
                # Giữ lại chỉ phần assistant thực sự, bỏ mọi metadata/hệ thống
                content = msg.content.strip()

                # Tách từng đoạn theo header (nếu có)
                parts = re.split(r"<\|start_header_id\|>.*?<\|end_header_id\|>", content)

                # Lấy phần cuối cùng (assistant nói)
                last_part = parts[-1] if parts else content

                # Xoá các tag còn sót lại
                clean_text = re.sub(r"<\|.*?\|>", "", last_part).strip()

                final_response = clean_text  # Ghi đè để lấy đoạn mới nhất
                print(final_response)

    print(final_response)
    return jsonify({"response": final_response})
@app.route('/chatrag', methods=['POST'])
def chatbotrag():
    data = request.get_json()
    user_input = data.get("user_input", "")
    print(user_input)
    answer = genai_chain.invoke(user_input)
    print(answer)
    return jsonify({"response": answer})
def answer_question(history, question):
    """
    This function takes the chat history and the new question as input,
    generates an answer, and updates the chat history.
    """
    # Generate answer using RAG model
    answer = genai_chain.invoke(question)
    
    # Append question and answer to history
    history.append((question, answer))
    return history


def run_flask():
    # Tắt debug để tránh lỗi signal trong thread phụ
    app.run(port=5000, debug=False, use_reloader=False)  # Thay đổi cổng từ 5555 sang 5000
if __name__ == '__main__':
    gen_doc = r"D:\NLP\medicalbot\COVID-19 Fact Sheet.pdf"
    # Build the RAG chain with LLM and document
    genai_chain = build_rag(llm, data_dir=gen_doc, data_type="pdf") # llm= None


    app.run(port=5000, debug=True, use_reloader=False) 


