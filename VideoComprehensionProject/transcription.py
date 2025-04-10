import io
from langchain_community.document_loaders import YoutubeLoader
import requests
from bs4 import BeautifulSoup

def get_video_title(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text)
    
    link= soup.find_all(name="title")[0]
    title = str(link)
    title = title.replace("<title>","")
    title = title.replace("</title>","")
    
    return title

def transcribe(video_info):
    return video_info[0].page_content

def save_transcription(filename, title, url, transcription):
    content = f"""Informações do Vídeo:

Título: {title}
URL: {url}

Transcrição: {transcription}
                """
    with io.open(filename, "w", encoding="utf-8") as file:
        file.write(content)

def main():
    video_url = "https://www.youtube.com/watch?v=II28i__Tf3M"
    video_title = get_video_title(video_url)

    video_loader = YoutubeLoader.from_youtube_url(
        video_url,
        language=["pt", "pt-BR", "en"],
    )

    complete_info = video_loader.load()
    transcription = transcribe(complete_info)

    save_transcription("transcription.txt", video_title, video_url, transcription)
    print("Transcrição salva com sucesso!")
    
if __name__ == "__main__":
    main()