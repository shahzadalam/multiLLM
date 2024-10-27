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
import gradio as gr
import whisper
import scipy.io.wavfile as wavfile
import numpy as np
import soundfile as sf

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


# Define a custom prompt template that focuses on getting the latest response
prompt_template = """The following is a conversation between a human and an AI assistant. Apart from current conversation dont include any other history.
Current conversation:
{history}
Human: {input}
AI:"""

PROMPT = PromptTemplate(
    input_variables=["history", "input"],
    template=prompt_template
)


def chooseModel(selected_option):
        if selected_option == "ChatGPT":
            llm = ChatOpenAI(model_name="gpt-3.5-turbo")
        elif selected_option == "Mistral":
            llm=HuggingFaceHub(repo_id="mistralai/Mistral-7B-Instruct-v0.2",model_kwargs={"temperature": 0.1,"top_p":0.9 })
        elif selected_option == "Llama":
            llm=HuggingFaceHub(repo_id="microsoft/Phi-3-mini-4k-instruct",model_kwargs={"temperature": 0.1})

        conversation = ConversationChain(
    llm=llm,
    prompt=PROMPT,
    verbose=False,
    memory=ConversationBufferMemory()
)

        return conversation
        
model = whisper.load_model("small")

# Function to process audio, convert it to text, and send to ChatGPT
def transcribe_and_ask_gpt(audio_tuple):
    sample_rate,audio_array = audio_tuple
            
    audio_path = "temp_audio.wav"
    sf.write(audio_path, audio_array, sample_rate)
    
    # Transcribe audio using Whisper
    result = model.transcribe(audio_path)
    text_input = result["text"]
    
    # Call ChatGPT API with transcribed text
    response = openai.completions.create(
        model="gpt-3.5-turbo-instruct",  # or "gpt-3.5-turbo" for GPT-3.5
        prompt=text_input,
        max_tokens=100
    )
    
    # Return GPT response
    gpt_output = response.choices[0].text.strip()
    print(gpt_output)
    
    return text_input, gpt_output

# Create the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Multi LLM Chatbot ")
    
    with gr.Row():
        option_select = gr.Radio(label="Select an Option", 
                                  choices=["ChatGPT", "Mistral", "Llama"], 
                                  value="ChatGPT")  # Default selection

    user_input = gr.Textbox(label="Your Message", placeholder="Type your message here...")
    
    # Output will show both user messages and bot responses
    chatbox = gr.Chatbot(label="Chat History")

    # Create a button to send the message
    submit_btn = gr.Button("Send")

    def update_chat(selected_option, user_message, chat_history):
        history = "\n".join([f"Human: {msg[0]}\nAssistant: {msg[1]}" for msg in chat_history])
        conversation = chooseModel(selected_option)
        bot_response = conversation.predict(input=user_message,history=history)
        chat_history.append((user_message, bot_response))
        return chat_history

    submit_btn.click(update_chat, 
                      inputs=[option_select, user_input, chatbox], 
                      outputs=chatbox)
    
    # Define Gradio interface with audio input and text output
    gr_interface = gr.Interface(
    fn=transcribe_and_ask_gpt,
    inputs=gr.Audio(type="numpy"),  # Removed 'source' argument
    outputs=[gr.Textbox(label="Transcribed Text"), gr.Textbox(label="ChatGPT Response")],
    title="Speech to ChatGPT",
    description="Speak into the microphone, and the text will be sent to ChatGPT."
    )



# Launch the app
demo.launch()



