from pydub import AudioSegment
import os, shutil, math

def extract_voice(file_name:str) -> None:
    audio = AudioSegment.from_wav(f'./audio/{file_name}.wav')
    if not os.path.exists(f'./audio/temp'): os.mkdir(f'./audio/temp')

    # split audio into chunks of 10 mins
    for t in range(math.ceil(len(audio)/600000)):
        newAudio = audio[t*600000 : (t+1)*600000]
        newAudio.export(f'./audio/temp/{file_name}_{t}.wav', format="wav")

        # Extract the human voice
        os.system(f"spleeter separate ./audio/temp/{file_name}_{t}.wav -o ./audio/temp/")
        
        if os.path.exists(f'./audio/temp/{file_name}_{t}/vocals.wav'):
            shutil.move(f'./audio/temp/{file_name}_{t}/vocals.wav', f'./audio/temp/{file_name}_{t}.wav')
        shutil.rmtree(f'./audio/temp/{file_name}_{t}')

    # combine sounds
    file_list = os.listdir(f'./audio/temp')
    combined = AudioSegment.empty()
    for file in file_list:
        combined += AudioSegment.from_file(f'./audio/temp/{file}', format="wav")

    # simple export
    combined.export(f'./audio/{file_name}.wav', format="wav")
    shutil.rmtree(f'./audio/temp')

