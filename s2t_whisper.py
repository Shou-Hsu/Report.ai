import os, re
from VAD import extract_voice

def download_from_vimeo(url:str) -> str:
    from vimeo_downloader import Vimeo
    import moviepy.editor as mp

    vimeo = Vimeo(url)   
    file_name = re.sub(r'[^\w\s]', '', vimeo._get_meta_data()[0]['title'].replace(' ', '_'))
    vimeo.best_stream.download(download_directory='./audio', filename=file_name)

    # Convert the video to WAV format
    clip = mp.AudioFileClip(f"./audio/{file_name}.mp4", fps=16000)
    clip.write_audiofile(f"./audio/{file_name}.wav")
    os.remove(f"./audio/{file_name}.mp4")

    return f"./audio/{file_name}.wav"

def download_from_youtube(url:str) -> str:
    import moviepy.editor as mp
    from pytube import YouTube

    # Download the video
    yt = YouTube(url, use_oauth=True, allow_oauth_cache=True)
    print('Start downloading')
    stream = yt.streams.filter(only_audio=True).first()
    stream.download()
    file_name = re.sub(r'[^\w\s]', '', yt.title).replace(' ', '_')
    os.rename(stream.default_filename, f"./audio/{file_name}.mp4")

    # Convert the video to WAV format
    clip = mp.AudioFileClip(f"./audio/{file_name}.mp4", fps=16000)
    clip.write_audiofile(f"./audio/{file_name}.wav")

    os.remove(f"./audio/{file_name}.mp4")
    print('Downloading is complete')

    return f"./audio/{file_name}.wav"

def remove_silence(file_name:str) -> None:
    from pydub import AudioSegment
    from pydub.silence import split_on_silence
    
    print('Strat remove the silence of audio')

    # Reading and splitting the audio file into chunks
    sound = AudioSegment.from_file(f'./audio/{file_name}.wav', format = 'wav')
    audio_chunks = split_on_silence(sound
                                ,min_silence_len = 100
                                ,silence_thresh = -45
                                ,keep_silence = 50
                            )

    # Putting the file back together
    combined = AudioSegment.empty()
    for chunk in audio_chunks:
        combined += chunk
    combined.export(f'./audio/{file_name}.wav', format = 'wav')
    print('Remove the silence is complete')

def punctuation_zh(content):
    from transformers import pipeline
    import math

    result = list()
    pipe = pipeline("token-classification", model="raynardj/classical-chinese-punctuation-guwen-biaodian")
    for i in range(math.ceil(len(content)/500)):
        result.extend([(j.get('entity'), j.get('end') + 500*i)  for j in pipe(content[i*500:(i+1)*500])])

    for i in range(len(result)-1,-1,-1):
        content = content[:result[i][1]] + result[i][0]+ content[result[i][1]:] 
    return content

def speech2text(file_path:str, model_name:str="tiny", extraction:bool=False) -> dict:
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
    from utils import translate_chinese, llm
    from langdetect import detect_langs
    from pydub import AudioSegment
    import json, torch

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    file_name = file_path.split('/')[-1].split('.')[0]
    file_type = file_path.split('/')[-1].split('.')[1]
    if file_type not in ["wav", "mp3"]: raise ValueError('Please make sure the audio is "wav" or "mp3"')

    # convert mp3 to wav
    if file_type == "mp3":
        audio = AudioSegment.from_mp3(f'./audio/{file_name}.mp3')
        audio.export(f'./audio/{file_name}.wav', format="wav")
        os.remove(f'./audio/{file_name}.mp3')

    # extract voice from audio
    if extraction: extract_voice(file_name)

    # remove the silence of audio
    remove_silence(file_name)
    
    # convert audio to text
    print('Start convert audio to text with timestamp')
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
    f'openai/whisper-{model_name}', torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True).to(device)

    processor = AutoProcessor.from_pretrained(f'openai/whisper-{model_name}')

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,
    )

    result = pipe(f'./audio/{file_name}.wav')
    content = result.get('text')
    language = str(detect_langs(content)[0]).split(':')[0]

    # add punctuation in chinese
    if language.__contains__('zh-cn'): 
        content = translate_chinese(llm, content)
        content = punctuation_zh(content)
        language = 'zh-tw'

    if language.__contains__('zh-tw'):
        content = punctuation_zh(content) 

    # save the transcript
    with open(f'./transcript/{file_name}.json', 'w', encoding='utf-8') as f: json.dump(result, f, ensure_ascii=False)
    with open(f'./transcript/{file_name}.txt', 'w', encoding='utf-8') as f: f.write(content)

    print('Converting is complete')
    return file_name, language