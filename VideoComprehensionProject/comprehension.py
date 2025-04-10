import os
import getpass
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.llms import HuggingFaceHub
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import transcription

def model_hf_hub(model="meta-llama/Meta-Llama-3-8B-Instruct", temperature=0.1):
    return HuggingFaceHub(
        repo_id=model,
        model_kwargs={
            "temperature": temperature,
            "return_full_text": False,
            "max_new_tokens": 1024,
            "cache": False
        }
    )

def model_openai(model="gpt-4o-mini", temperature=0.1):
    return ChatOpenAI(model=model, temperature=temperature)

def model_ollama(model="phi3", temperature=0.1):
    return ChatOllama(model=model, temperature=temperature)

def get_video_transcription(url):
    loader = YoutubeLoader.from_youtube_url(
        url,
        language=["pt", "pt-BR", "en"]
    )
    docs = loader.load()
    return transcription.transcribe(docs)

def select_llm():
    while True:
        choice = input(
            "Enter a number for the model:\n"
            "1 - HuggingFace Model\n"
            "2 - OpenAI Model\n"
            "3 - Ollama Model\n"
            "0 - Exit\n"
            "Your Choice: "
        )

        if choice == "1":
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = getpass.getpass("Enter your Hugging Face token: ")
            return model_hf_hub(), choice
        elif choice == "2":
            return model_openai(), choice
        elif choice == "3":
            return model_ollama(), choice
        elif choice == "0":
            print("Exit")
            exit()
        else:
            print("Enter a valid number")

def build_prompt(choice):
    system_prompt = (
        "Você é um assistente virtual prestativo e deve responder a uma consulta com base "
        "na transcrição de um vídeo, que será fornecida abaixo."
    )

    inputs = "Consulta: {consulta} \n Transcrição: {transcription}"

    if choice == "1":
        user_prompt = (
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n"
            f"{inputs}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        )
    else:
        user_prompt = inputs

    return ChatPromptTemplate.from_messages([("system", system_prompt), ("user", user_prompt)])

def main():
    video_url = "https://www.youtube.com/watch?v=II28i__Tf3M"
    transcription_text = get_video_transcription(video_url)

    llm, choice = select_llm()
    prompt_template = build_prompt(choice)

    chain = prompt_template | llm | StrOutputParser()
    result = chain.invoke({
        "transcription": transcription_text,
        "consulta": "resuma"
    })

    print("\n🔎 Resposta:")
    print(result)

if __name__ == "__main__":
    main()