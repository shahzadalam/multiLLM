import os
import openai
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
from langchain.chains import ConversationChain

# Hugging face API Token
#HUGGINGFACEHUB_API_TOKEN = userdata.get('HF_TOKEN')
#os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN

# OpenAI API key
#OPENAI_API_KEY = userdata.get('OA_API')
#os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Read OpenAI key from Codespaces Secrets
api_key = os.environ['OA_API']            
os.environ['OPENAI_API_KEY'] = api_key
openai.api_key = os.getenv('OPENAI_API_KEY')

# Load Model
llm = ChatOpenAI(model_name="gpt-3.5-turbo")

response = llm.invoke('Hi!')
print(f'Actual response: {response.content}\n')
print('Raw response:',response)

