import os

import numpy as np
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms.llamafile import Llamafile
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnablePick
from pydantic import BaseModel
from tensorflow.keras.models import load_model
import logging

load_dotenv()
logging.basicConfig(format='%(levelname)s - %(asctime)s: %(message)s', datefmt='%H:%M:%S', level=logging.INFO)
app = FastAPI()
model = load_model("bp.keras")



MONGO_URI = os.getenv("MONGODB_ATLAS_CLUSTER_URI")
DB_NAME = os.getenv("MONGODB_NAME")
COLLECTION_NAME = os.getenv("MONGODB_COLLECTION_NAME")
ATLAS_VECTOR_SEARCH_INDEX_NAME = "medical_info_index"

embeddings = HuggingFaceEmbeddings(model_name="thenlper/gte-large")

vector_search = MongoDBAtlasVectorSearch.from_connection_string(
    MONGO_URI,
    DB_NAME + "." + COLLECTION_NAME,
    embeddings,
    index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME
)

llm = Llamafile()
llm.base_url = os.getenv("LLM_URL")
rag_prompt = PromptTemplate.from_template("You are an assistant for question-answering tasks. Use the following "
                                          "pieces of retrieved context to answer the question. If you don't know the "
                                          "answer, just say that you don't know. Use three sentences maximum and keep "
                                          "the answer concise.Question: {question} \nContext: {context}")



def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


chain = (
        RunnablePassthrough.assign(context=RunnablePick("context") | format_docs)
        | rag_prompt
        | llm
        | StrOutputParser())


class BPVitals(BaseModel):
    age: float
    sex: int
    bmi: float
    systolic_bp: list[float]
    diastolic_bp: list[float]


@app.post("/bp")
async def predict_bp(bp_vitals: BPVitals):
    context_data = [bp_vitals.age, bp_vitals.sex, bp_vitals.bmi]
    time_series_data = [bp_vitals.systolic_bp, bp_vitals.diastolic_bp]
    context_data = np.array([context_data], dtype=float)
    time_series_data = np.array(time_series_data, dtype=float).reshape(-1, 14, 2)
    logging.info(f'Time Series :{time_series_data}')
    print(f'Context Data:{context_data}')
    results = model.predict(x=[context_data, time_series_data])
    logging.info(f"Results:{results}")
    question = f'The Blood pressure measured for 7 days is Systolic:{results[0][0:7].tolist()} and Diastolic:{results[0][0:14].tolist()}.Is my Blood pressure Bad? If it is bad what actions do you recommend?'
    docs = vector_search.similarity_search(question)
    thoughts = chain.invoke(
        {"context": docs, "question": question})
    logging.info(f"Alton {thoughts}")
    return {"future_systolic_bp": results[0][0:7].tolist(), "future_diastolic_bp": results[0][7:14].tolist(),"thoughts":thoughts}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)