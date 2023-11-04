from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from utils import fuzzy_match, convert_json
import os, difflib, re, json

class divide_article():
    def __init__(self, file_name:str, original_language:str, chunk_size:int) -> None:
        from utils import llm
        self.file_name = file_name
        self.chunk_size = chunk_size 
        self.original_language = original_language
        self.llm = llm
        self.llm.temperature=0.0

    def _get_timestamp_list(self, article_timestamp:dict) -> list:
        timestamp_list = list()
        for segment in article_timestamp.get('segments'):
            for word in segment.get('words'):
                texts = ''.join(re.split(r"\W+", word.get('text').strip()))

                for text in texts: 
                    start = word.get('start')
                    end = word.get('end')
                    timestamp_list.append((text, start, end))
        return timestamp_list

    def _add_timestamp(self, paragraphs:list) -> dict:      
        with open(f'./transcript/{self.file_name}.json') as f:
            timestamp_list = self._get_timestamp_list(json.load(f))
        result, subtopics = dict(), list()
        for paragraph in paragraphs.get('Subtopics'):
            # seperate transcript to word list
            primer = ''.join(re.split(r"\W+", paragraph.get('transcript').strip()))
            word_list = [word for word in primer if word]
            start, end = 'undifine', 'undifine'
            index_w = 0

            # fit the timestamp to the paragraph
            for timestamp in timestamp_list:
                if index_w == len(word_list): break
                if timestamp[0] == word_list[index_w]:
                    if start == 'undifine': start = int(timestamp[1])
                    end = int(timestamp[2])
                    index_w += 1
                else: 
                    start, end = 'undifine', 'undifine'
            paragraph['timestamp'] = {"start":start, "end":end}
            subtopics.append(paragraph)
        result['Subtopics'] = subtopics
        return result

    def _add_transcript(self) -> dict:       
        with open(f'./transcript/{self.file_name}.txt') as f:
            transcript = ''.join(f.readlines())

        result, subtopics = dict(), list()
        index_list = [['start', 0]]
        # divide the transcript by punctuation
        source = re.split(r'[:.?!，。]', transcript)
        for paragraph in self.draft:
            subtopic = paragraph.get('Subtopic')
            primer = re.split(r'[:.?!，。]', paragraph.get('Transcript').strip())[0]

            # fuzzy match the primer and transcript
            matched = fuzzy_match(source, primer, cutoff=0.1)
            if matched: 
                if transcript.find(matched[0], index_list[-1][1]) == -1: index_list.pop()
                index_list.append((subtopic, transcript.find(matched[0], index_list[-1][1])))

        index_list.append(('end', len(transcript)))
        # fulfill transcript
        for i in range(1, len(index_list)-1):
            if index_list[i+1][1] - index_list[i][1] > 10:
               subtopic_dict = dict()
               subtopic_dict['subtopic'] = index_list[i][0]
               subtopic_dict['transcript'] = transcript[index_list[i][1]:index_list[i+1][1]]
               subtopics.append(subtopic_dict)

        result['Subtopics'] = subtopics
        return result
       
    def _divide_by_subtopics(self) -> dict:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain.chains.summarize import load_summarize_chain

        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size = self.chunk_size,
                            chunk_overlap  = 0,
                            length_function = len)
        
        with open(f'./transcript/{self.file_name}.txt') as f:
            transcript = ''.join(f.readlines())
        split_text = text_splitter.split_text(transcript)

        # Define prompt
        prompt_template = "###The following text is part of an article, and your task is to identify and categorize its subtopics.\
                            Please ensure that you do not overly fragment the content, and \
                            that each subtopic contains a sufficient amount of information.\
                            Begin by identifying the subtopics within the text.\
                            Keep the context entirely unmodified and refrain from extending it in any way.\
                            Divide the given text into separate contexts based on the identified subtopics."\
                            f"Extract the first sentence from each context as a transcript in {self.original_language}."\
                            f"Discard the remainder of the transcript, retaining only the first sentence in {self.original_language}."\
                            "Fulfill the format below: \n\
                            Subtopic: \n\
                            Transcript: \n\
                            ###{text}\n"
        
        prompt = PromptTemplate.from_template(prompt_template)

        # Define LLM chain
        llm_chain = LLMChain(llm=self.llm, prompt=prompt)

        # Define StuffDocumentsChain
        stuff_chain = StuffDocumentsChain(  
            llm_chain=llm_chain, document_variable_name="text", input_key='language')

        # divide article
        output = list()
        print('Dividing the content')
        for chunk in split_text:
            doc = Document(page_content=chunk, metadata={"source": self.file_name})
            out = stuff_chain.run([doc]).strip()
            output.append(convert_json(out, ['Subtopic', 'Transcript']))

        self.draft = output

    def run(self):       
       # divide article 
        self._divide_by_subtopics()
       
        # # fulfill transcript
        article_full = self._add_transcript()

        # add timestamp, base on whisper result 
        if os.path.exists(f'./transcript/{self.file_name}.json'):
            article_full = self._add_timestamp(article_full)
        
        # save result
        with open(f'./summary/{self.file_name}.json', 'w', encoding='utf-8') as f:
            json.dump(article_full, f, ensure_ascii=False)

        return article_full