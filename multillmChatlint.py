import os
import openai
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub
import pandas as pd
from langchain.chains import ConversationChain 
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import chainlit as cl
from chainlit.input_widget import Select

# Variable to track if the option has been used
option_used = False
model = "Mistral"

# Hugging face API Token
HUGGINGFACEHUB_API_TOKEN = os.environ['HF_TOKEN']
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN

# Read OpenAI key from Codespaces Secrets
api_key = os.environ['OA_API']            
os.environ['OPENAI_API_KEY'] = api_key
openai.api_key = os.getenv('OPENAI_API_KEY')

# Choose the model

Model_input = input("Please select the model 1. ChatGPT 2. Mistral :") 

if(Model_input=="1"):
    print("You have selected ChatGPT \n")
    llm = ChatOpenAI(model_name="gpt-3.5-turbo")
else:
    print("You have selected Mistral \n")
    llm=HuggingFaceHub(repo_id="mistralai/Mistral-7B-Instruct-v0.2")


# Define a custom prompt template that focuses on getting the latest response
prompt_template = """The following is a conversation between a human and an AI assistant.

Current conversation:
{history}
Human: {input}
AI:"""

PROMPT = PromptTemplate(
    input_variables=["history", "input"],
    template=prompt_template
)

conversation = ConversationChain(
    llm=llm,
    prompt=PROMPT,
    verbose=False,
    memory=ConversationBufferMemory()
)

def get_clean_response(user_input):
    response = conversation.predict(input=user_input)
    # Remove any prefixes if they appear
    response = response.replace('Assistant:', '').replace('AI:', '').strip()
    return response
"""
while True:
    # Get user input
    user_input = input("User: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    # Get the assistant's response
    
    response = get_clean_response(user_input)
    print("\nAssistant:", response)
"""

@cl.on_chat_start            # for actions to happen when the chat starts
async def main():
    await cl.Message(content=f"Welcome to Chat Assistant").send()

@cl.on_message               # for actions to happen whenever user enters a message
async def main(message: cl.Message):
    response = get_clean_response(message.content)    #clf_chain.invoke({"request": message.content})
    await cl.Message(content=f"Response: \n{[str(response)]}",).send()

