from dotenv import load_dotenv
import docx, difflib ,re , os
import validators, shutil

def convert_json(txt:str, item_list:list) -> str:
    txt = txt.replace('\n', '').replace('#', '')

    output = dict()
    for i in range(len(item_list)):
        start = txt.lower().find(item_list[i].lower() + ':')

        if i != len(item_list) - 1: 
            end = txt.lower().find(item_list[i+1].lower() + ':')
        else:
            end = len(txt)

        output[item_list[i]] = txt[start + len(item_list[i]) + 1 : end].strip()

    return output

def fuzzy_match(source:str, target:str, cutoff:float=0.1) -> float:
    if type(source) == str: source = re.split(r'[:.?!，。]', source)

    return difflib.get_close_matches(target, source, n=1, cutoff=cutoff)

def validation_and_filetype_check(file_path:str, output_dir:str='./docx') ->str:

    if not os.path.exists('./transcript'): os.mkdir('./transcript')
    if not os.path.exists('./summary'): os.mkdir('./summary')
    if not os.path.exists(output_dir): os.mkdir(output_dir)
    if not os.path.exists('./audio'): os.mkdir('./audio')

    # validate input is url or not
    if validators.url(str(file_path)): return 'url', file_path

    # validate file path is existed or not
    if os.path.exists(file_path): 
        file_name = file_path.split('/')[-1].split('.')[0]
        # validate input is txt or not
        if file_path.endswith('.txt'): 
            # copy file to default folder
            if not os.path.exists(f'./transcript/{file_name}.txt'):
                shutil.copyfile(file_path, f'transcript/{file_name}.txt')
            return 'transcript', file_name
        
        # validate input is wav or not
        elif file_path.endswith('.wav'): 
            # copy file to default folder
            if not os.path.exists(f'./audio/{file_name}.wav'):
                shutil.copyfile(file_path, f'audio/{file_name}.wav')
            return 'audio', file_name
        else:
            raise ValueError(f'Please check input type is url or txt or wav')
        
    else: raise ValueError(f'Please check {file_path} is existed or not')

def detect_language(file_path:str) -> str:
    from langdetect import detect

    with open(file_path,'r') as f:
        text = ''.join(f.readlines())
    return detect(text)

def get_items(type:str):
    if type == 'individuel':
        with open('./template/individuel.txt') as f: lines = f.readlines()
    elif type == 'general':
        with open('./template/general.txt') as f: lines = f.readlines()
    else: raise ValueError('type must be "individuel" or "general"')

    item = ''.join(lines)
    item_list = ''.join(lines).split(',') 
    item_list = [items.strip() for items in item_list]
    item_format = ''
    for i in item_list: item_format += f'{i}:'

    return item_list, item, item_format

def add_hyperlink(paragraph, text, url):
    # This gets access to the document.xml.rels file and gets a new relation id value
    part = paragraph.part
    r_id = part.relate_to(url, docx.opc.constants.RELATIONSHIP_TYPE.HYPERLINK, is_external=True)

    # Create the w:hyperlink tag and add needed values
    hyperlink = docx.oxml.shared.OxmlElement('w:hyperlink')
    hyperlink.set(docx.oxml.shared.qn('r:id'), r_id, )

    # Create a new run object (a wrapper over a 'w:r' element)
    new_run = docx.text.run.Run(docx.oxml.shared.OxmlElement('w:r'), paragraph)
    new_run.text = text

    # Join all the xml elements together
    hyperlink.append(new_run._element)
    paragraph._p.append(hyperlink)

    return hyperlink

def divide_audio(file_name:str, subtopic:str, start:float, end:float) -> str:
    from pydub import AudioSegment

    if not os.path.exists(f'./audio/{file_name}'): os.mkdir(f'./audio/{file_name}')

    myaudio = AudioSegment.from_file(f'./audio/{file_name}.wav', "wav")
    sliced_audio = myaudio[start*1000:end*1000]
    sliced_audio.export(f'./audio/{file_name}/{subtopic}.wav', format="wav")
    absolute_path = os.path.abspath(f'./audio/{file_name}')
    return absolute_path

def get_file_list(file_path:str) -> list:
    with open(file_path) as f: lines = ''.join(f.readlines())
    
    return [line.strip() for line in lines.split(',') if line]

def credential_validation(vectorDB:str=False, temperature:float=0.1) -> None:
    from langchain.embeddings.openai import OpenAIEmbeddings
    from langchain.chat_models import AzureChatOpenAI
    from langchain.chat_models import ChatOpenAI

    load_dotenv()
    # validate llm
    global llm, pinecone, embeddings

    if os.getenv('AZURE_OPENAI_API_KEY') and os.getenv('AZURE_OPENAI_API_BASE') and os.getenv('AZURE_OPENAI_API_VERSION') and os.getenv('AZURE_DEPLOYMENT_NAME'):
        llm = AzureChatOpenAI(openai_api_key=os.getenv('AZURE_OPENAI_API_KEY'),
                              openai_api_base=os.getenv('AZURE_OPENAI_API_BASE'),
                              openai_api_version=os.getenv('AZURE_OPENAI_API_VERSION'),                            
                              deployment_name=os.getenv('AZURE_DEPLOYMENT_NAME'), 
                              temperature=temperature, 
                              request_timeout=240)
        
        embeddings = OpenAIEmbeddings(openai_api_key=os.getenv('AZURE_OPENAI_API_KEY'),
                                      openai_api_base=os.getenv('AZURE_OPENAI_API_BASE'),
                                      openai_api_version=os.getenv('AZURE_OPENAI_API_VERSION'),  
                                      deployment=os.getenv('EMBEDDING_DEPLOYMENT_NAME'))

        print('Initial AzureOpenAI')
    elif os.getenv('OPENAI_API_KEY'):
        llm = ChatOpenAI(
            openai_api_key=os.getenv('OPENAI_API_KEY'),
            model_name=os.getenv('MODEL_NAME'), 
            temperature=temperature, 
            request_timeout=240
            )
        embeddings = OpenAIEmbeddings()
        print('Initial OpenAI')
    else:
        raise Exception('Please provide OPENAI_API_KEY')


    # validate pinecone
    if vectorDB == 'pinecone':
        import pinecone
        if os.getenv('PINECONE_API_KEY') and os.getenv('PINECONE_ENV'):
            pinecone.init(environment=os.getenv('PINECONE_ENV'))
            print('Initial Pinecone')
        else:
            raise Exception('Please provide PINECONE_API_KEY and PINECONE_ENV')
        
