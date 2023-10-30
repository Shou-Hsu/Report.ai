from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from utils import convert_json, get_items
from tqdm import tqdm
import json


class generate_summary():
    def __init__(self, file_name:str, language:str, chunk_size:int, output_dir:str) -> None:
        from utils import llm
        self.file_name = file_name
        self.chunk_size = chunk_size 
        self.language = language
        self.output_dir = output_dir
        self.llm = llm

    def _get_general_summary(self, article_divided:dict) -> None:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain.chains.summarize import load_summarize_chain

        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size = self.chunk_size//2,
                            chunk_overlap  = 0,
                            length_function = len)
        
        # load transcript
        with open(f'./transcript/{self.file_name}.txt', 'r') as f:
            transcript = ''.join(f.readlines())
        split_text = text_splitter.split_text(transcript)

        item_list, items, item_format  = get_items('general')
        prompt_template = f"Write cosine {items} of the following:" """ ```{text}``` \n"""
        prompt = PromptTemplate.from_template(prompt_template)

        refine_template = (
            f"Your job is to produce a final streamline {items}."\
            f"We have provided an existing {items} up to a certain point:" """```{existing_answer}```\n"""
            f"We have the opportunity to refine {items}"
            """(only if needed) with some more context below.\n
            ------------\n
            ```{text}```\n
            ------------\n"""
            f"Given the new context, refine the original {items}"
            f"If the context isn't useful, return the origin {items}"
            f"Fulfill the format below: \n {item_format}"
    )
        refine_prompt = PromptTemplate.from_template(refine_template)

        chain = load_summarize_chain(
            llm=self.llm,
            chain_type="refine",
            question_prompt=prompt,
            refine_prompt=refine_prompt,
            return_intermediate_steps=False,
            input_key="input_documents",
            output_key="output_text"
        )
        print('Analysing general items')
        split_docs = [Document(page_content=text, metadata={"source": self.file_name}) for text in split_text]
        out = chain({"input_documents": split_docs}, return_only_outputs=True)['output_text']

        # convert to json
        output = convert_json(out, item_list)

        self.article_full = {**output, **article_divided}

    def _get_subtopic_summary(self) -> None:
        item_list, items, format = get_items('individuel')

        prompt_template = f"Find out or extract the {items} according to the information provided in the following text behind ###.\
                            Remember, you are being trained to find outor extract specific information without fail so you must \
                            ensure that they are all presented in the specified format below: \n"\
                            "Subtopic:-||-,"\
                            f'{format}'\
                            """```{text}```"""

        prompt = PromptTemplate.from_template(prompt_template)

        # Define LLM chain
        llm_chain = LLMChain(llm=self.llm, prompt=prompt, return_final_only=True)

        # Define StuffDocumentsChain
        stuff_chain = StuffDocumentsChain(  
            llm_chain=llm_chain, document_variable_name="text")
        
        print('Analysing subtopics')
        result = list()
        with tqdm(total=len(self.article_full.get('Subtopics'))) as pbar:
            for subtopic in self.article_full.get('Subtopics'):
                content = f"{subtopic.get('subtopic')} :  {subtopic.get('transcript').strip()}"
                doc = Document(page_content=content , metadata={"source": self.file_name})
                out = stuff_chain.run([doc])

                # convert to json
                output = convert_json(out, item_list)

                output['subtopic'] = subtopic.get('subtopic')
                doc = Document(page_content=subtopic.get('transcript') , metadata={"source": self.file_name})
                output['original transcript'] = subtopic.get('transcript')
                if subtopic.get('timestamp'):
                    output['timestamp']= [{'start': subtopic.get('timestamp').get('start')}, {'end': subtopic.get('timestamp').get('end')}]
                result.append(output)
                pbar.update(1)
            self.article_full.update({"Subtopics":result})
        with open(f'./summary/{self.file_name}.json', 'w', encoding='utf-8') as f: json.dump(self.article_full, f, ensure_ascii=False)
        print("Analysis completed")

    def _translate_chinese(self, doc:object) -> str:
        prompt_template = """你是一位資深的中文翻譯員，會將以下的英文內容以忠於原文不多加延伸跟解釋的方式翻譯成繁體中文，並且熟知 Azure 以及 OpenAI 對於內容審查的所有標準，因此當在你產生新的文字前，請先確認新產生的詞彙是否會違反 Azure 及 OpenAI 的審核標準，若有任何一點可能違反的機會，請立即停止文字生成並置換成其他相似但不違反的 Azure 及 OpenAI 政策的詞句。```{text}```"""
       
        prompt = PromptTemplate.from_template(prompt_template)
        llm_chain = LLMChain(llm=self.llm, prompt=prompt, return_final_only=True)

        stuff_translate_chain = StuffDocumentsChain(  
            llm_chain=llm_chain, document_variable_name="text")

        return stuff_translate_chain.run([doc])

    
    def _translate_convert_docx(self) -> None:
        from utils import add_hyperlink, divide_audio
        import docx, datetime

        # initial a docx
        document = docx.Document()

        # translate general info and convert in docx
        print('Start translation')
        items_list, _, _ = get_items('general')
        for item in items_list:
            content = self.article_full.get(item)
            if content:
                doc = Document(page_content=''.join(content).strip(), metadata={"source": self.file_name})
                output = self._translate_chinese(doc)
            else: output = '無'
            document.add_heading(item, level=1)
            document.add_paragraph(output)

        subtopics = self.article_full.get('Subtopics')
        with tqdm(total=len(subtopics)) as pbar:
            for subtopic in subtopics:
                content = subtopic.get("subtopic")
                if content:
                    doc = Document(page_content=''.join(content).strip(), metadata={"source": self.file_name})
                    output = self._translate_chinese(doc).strip()
                else: output = '無'
                insertion = document.add_heading(output, level=2)

                # add hyperlink
                if subtopic.get('timestamp') and isinstance(subtopic.get('timestamp')[0].get('start'), int) and isinstance(subtopic.get('timestamp')[1].get('end'), int):
                    start = subtopic.get('timestamp')[0].get('start')
                    end = subtopic.get('timestamp')[1].get('end')
                    subtopic_name = subtopic.get('subtopic')
                    # seperate audio by suntopics
                    absolute_path = divide_audio(self.file_name, subtopic_name, start, end)
                    add_hyperlink(insertion, f'{datetime.timedelta(seconds = int(start))}', f'file:///{absolute_path}/{subtopic.get("subtopic")}.wav')

                # translate individual item and convert in docx
                items_list, _, _ = get_items('individuel')
                for item in items_list:
                    content = subtopic.get(item)
                    if content:
                        doc = Document(page_content=''.join(content), metadata={"source": self.file_name})
                        output = self._translate_chinese(doc).replace('\n', ',')
                    else: output = '無'
                    document.add_heading(item, level=3)
                    document.add_paragraph(output)

                # add chinese transcript
                content = f'{subtopic.get("original transcript")}'
                doc = Document(page_content=content , metadata={"source": self.file_name})
                if self.language != 'zh':
                    document.add_heading('中文逐字稿', level=3)
                    document.add_paragraph(self._translate_chinese(doc).strip())
                document.add_heading('原文逐字稿', level=3)
                document.add_paragraph(subtopic.get('original transcript'))
                pbar.update(1)

                document.save(f'{self.output_dir}/{self.file_name}.docx')

    def run(self, article_divided:dict) -> None:
        # generate global and subtopic summary
        self._get_general_summary(article_divided)
        self._get_subtopic_summary()
        
        # Translate and convert json to docx
        self._translate_convert_docx()
