<div align="center">
<h1 align="center">
<img src="icon/Report.ai.jpeg" width="200" />
<br>Report.ai</h1>
<h3>◦ Empower Your Data with Report.ai</h3>
<h3>◦ Developed with the software and tools below.</h3>

<p align="center">
<img src="https://img.shields.io/badge/Langchain-FFC107.svg?style=flat-square&logo=Langchain&logoColor=black" alt="Langchain" />
<img src="https://img.shields.io/badge/OpenAI-412991.svg?style=flat-square&logo=OpenAI&logoColor=white" alt="OpenAI" />
<img src="https://img.shields.io/badge/Whisper-ECD53F.svg?style=flat-square&logo=Whisper&logoColor=black" alt="Whisper" />
<img src="https://img.shields.io/badge/Spleeter-3776AB.svg?style=flat-square&logo=Spleeter&logoColor=white" alt="Spleeter" />
<img src="https://img.shields.io/badge/Pinecone-007808.svg?style=flat-square&logo=Pinecone&logoColor=white" alt="Pinecone" />
</p>
<img src="https://img.shields.io/github/license/Shou-Hsu/Report.ai?style=flat-square&color=5D6D7E" alt="GitHub license" />
<img src="https://img.shields.io/github/last-commit/Shou-Hsu/Report.ai?style=flat-square&color=5D6D7E" alt="git-last-commit" />
<img src="https://img.shields.io/github/commit-activity/m/Shou-Hsu/Report.ai?style=flat-square&color=5D6D7E" alt="GitHub commit activity" />
<img src="https://img.shields.io/github/languages/top/Shou-Hsu/Report.ai?style=flat-square&color=5D6D7E" alt="GitHub top language" />
</div>

---

## 📖 Table of Contents
- [📖 Table of Contents](#-table-of-contents)
- [📍 Overview](#-overview)
- [📦 Features](#-features)
- [📂 repository Structure](#-repository-structure)
- [⚙️ Modules](#modules)
- [🚀 Getting Started](#-getting-started)
    - [🔧 Installation](#-installation)
    - [🤖 Running Report.ai](#-running-Report.ai)
    - [🧪 Quickstart](#-quickstart)
- [🛣 Roadmap](#-roadmap)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)
- [👏 Acknowledgments](#-acknowledgments)

---


## 📍 Overview

At Report.ai, our mission is clear: to empower you with a robust AI-driven reporting experience. We've moved beyond the limitations of traditional text length-based segmentation, opting for a smarter approach—semantic segmentation. This innovative method ensures unparalleled precision in identifying both overarching themes and nuanced details within your content.Moreover, we go the extra mile by offering a transcript and audio within each segment, providing you with a reliable reference point for a comprehensive understanding of your content.

### Download the audio from Youtube and convert to transcipt with timestamp
<a href="https://youtu.be/RSNWOGQTsns">
  <img src="https://img.youtube.com/vi/RSNWOGQTsns/maxresdefault.jpg" alt="Video Preview" width="80%">
</a>

### Analyze the content using customized template.
<a href="https://youtu.be/DBJqP350uu4">
  <img src="https://img.youtube.com/vi/DBJqP350uu4/maxresdefault.jpg" alt="Video Preview" width="80%">
</a>

### The output report
<a href="https://youtu.be/dxRgrQZRJPY">
  <img src="https://img.youtube.com/vi/dxRgrQZRJPY/maxresdefault.jpg" alt="Video Preview" width="80%">
</a>

---

## 📦 Features

<h3>1. Semantic Segmentation: </h3> Instead of relying on text length, Report.ai segments your reports by their meaning. This results in a more accurate breakdown of content, enhancing your understanding of the material.

<h3>2. Interactive Transcripts: </h3> Our reports go beyond mere text representation. Each semantic chunk is presented alongside an interactive transcript, allowing you to seamlessly navigate and reference the original audio segments.

<h3>3. Customizable Templates: </h3> We put the power of customization in your hands. Tailor your analysis with ease using our customizable templates, empowering you to extract insights that matter to you.

<h3>4. Multimedia Integration: </h3> Whether you're working with YouTube links, audio files in WAV format, or text transcripts in TXT format, we've got you covered. Report.ai seamlessly handles a variety of multimedia inputs, making your experience comprehensive and convenient.

<h3>5. Professional Database Support: </h3> For those seeking to establish a professional database, our repository provides seamless integration with Pinecone and Chroma. These advanced tools offer superior data management and retrieval capabilities, enhancing the value of your reporting efforts.

---


## 📂 Repository Structure

```sh
└── readme/
    ├── .env
    ├── VAD.py
    ├── divide.py
    ├── example/
    │   ├── WATCH_LIVE_Nvidia_Q2_Earnings_Call_NVDA
    │   └── batch.txt
    ├── main.py
    ├── requirements.txt
    ├── s2t_whisper.py
    ├── storage_vector.py
    ├── summarize.py
    ├── template/
    │   ├── general.txt
    │   └── individuel.txt
    └── utils.py

```

---


## ⚙️ Modules

<details closed><summary>Root</summary>

| File                                                                                   | Summary|
| ---                                                                                    | ---|
| [requirements.txt](https://github.com/Shou-Hsu/Report.ai/blob/main/requirements.txt)   | Providing a list of essential dependencies crucial for the proper functioning of the code.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| [.env](https://github.com/Shou-Hsu/Report.ai/blob/main/.env)                           | The `.env` file serves as a repository for configuration settings pertaining to various APIs, encompassing those from OpenAI, Azure OpenAI, and Pinecone. Within this file, you'll find essential information like API keys, model names, and storage configurations.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| [utils.py](https://github.com/Shou-Hsu/Report.ai/blob/main/utils.py)                   | Within the `utils.py` file, you'll discover a comprehensive array of utility functions. These functions are versatile and span various essential tasks, including: fuzzy_match: For performing fuzzy string matching. validate_filetype: Ensuring file type validation. detect_language: Detecting the language of a text file. get_items: Extracting items from template files. add_hyperlink: Adding hyperlinks within Word documents. divide_audio: Slicing audio files into segments. get_file_list: Retrieving lists of file paths.                                                                                                                                                                                                                  |
| [summarize.py](https://github.com/Shou-Hsu/Report.ai/blob/main/summarize.py)           | The `summarize.py` script is dedicated to generating summaries based on the templates found in template/general.txt and template/individual.txt. These summaries can be translated, if required, and then transformed into Microsoft Word document format (.docx). Throughout this process, the document is enriched with hyperlinks and additional contextual details.                                                                                                                                                                                                                                                                                                                                                                                   |
| [s2t_whisper.py](https://github.com/Shou-Hsu/Report.ai/blob/main/s2t_whisper.py)       | The `s2t_whisper.py` provides functionalities to download YouTube videos, extract the audio, remove silence, convert speech to text with timestamp, and add punctuation for Chinese content. The resulting transcript is saved in both JSON and TXT format.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| [VAD.py](https://github.com/Shou-Hsu/Report.ai/blob/main/VAD.py)                       | The `VAD.py` is used to extract human voice from an audio file. It splits the audio into chunks of 10 minutes, exports each chunk as a separate file, and extracts the human voice using the Spleeter library. The extracted vocals are then combined into a single audio file.                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| [divide.py](https://github.com/Shou-Hsu/Report.ai/blob/main/divide.py)                 | The `divide.py` is to that divides an article into subtopics based on its transcript. The class has several private methods: `_string_cleaner` cleans the input string, `_get_timestamp_list` extracts timestamps from a JSON file, `_add_timestamp` adds timestamps to subtopics, `_add_transcript` add the transcript into subtopics, and `_divide_by_subtopics` uses language models to divide the article into chunks.                                                                                                                                                                                                                                                                                                                                |
| [main.py](https://github.com/Shou-Hsu/Report.ai/blob/main/main.py)                     | The `main.py` is a versatile script designed for file analysis and summary generation. It offers extensive flexibility by accepting various command-line arguments, including: `File Path`: To specify the file for analysis. `Chunk Size`: Allowing you to define the size of text segments. `Temperature of Language Model`: To fine-tune the behavior of the language model. `Batch Mode`: Enabling you to indicate whether the script should run in batch mode. `Report Generation`: Providing the option to create a report. `Vector Database Selection`: Allowing you to choose between Pinecone and Chroma vector databases. `ASR (Automatic Speech Recognition) Model`: For selecting the appropriate ASR model to be used.                       |
| [storage_vector.py](https://github.com/Shou-Hsu/Report.ai/blob/main/storage_vector.py) | The `storage_vector.py` script offers two essential functions: pinecone_storage and chroma_storage, both designed to facilitate the storage of results in a vector database.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |

</details>

<details closed><summary>template</summary>

| File                                                                                               | Summary                                                                                                                                                                                                                                                                                                                                                                                                          |
| ---                                                                                                | ---                                                                                                                                                                                                                                                                                                                                                                                                              |
| [individuel.txt](https://github.com/Shou-Hsu/Report.ai/blob/main/template/individuel.txt) | The content of the `individuel.txt` provides items that are analyzed within each subtopic.                                             |
| [general.txt](https://github.com/Shou-Hsu/Report.ai/blob/main/template/general.txt)       | The content of the `general.txt` provides items that are analyzed within whole transcript.                                             |

</details>

<details closed><summary>Example</summary>

| File                                                                                                                                                                      | Summary                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| ---                                                                                                                                                                       | ---                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| [batch.txt](https://github.com/Shou-Hsu/Report.ai/blob/main/example/batch.txt)                                                                                   | The `batch.txt` file, is used to facilitate the processing of multiple files. It achieves this by listing the file paths, separated by commas, to indicate which multiple files are to be processed sequentially.         |
| [WATCH_LIVE_Nvidia_Q2_Earnings_Call_NVDA.txt](https://github.com/Shou-Hsu/Report.ai/blob/main/example/WATCH_LIVE_Nvidia_Q2_Earnings_Call_NVDA.txt)               | `WATCH_LIVE_Nvidia_Q2_Earnings_Call_NVDA.txt`, contains a transcript of NVIDIA's Q2 2023 financial results and Q&A webcast.                                                                                               |

</details>

---

## ⚙️ Configuration

| Short Flag | Long Flag     | Description                                         |   Type    | Status |
| -----------| --------------|-----------------------------------------------------|-----------|--------|
| - o        | --output_dir  | Setting the output directory for the report, Default is ./docx  | 	string   | Option |
| - c        | --chunk       | Setting chunk size for analysis. recommendatin (GPT-3.5: 10000 in en, 2000 in zh, GPT-4: 18000 in en, 3600 in zh), Default is 2000  | 	String  | Option |
| - t        | --temperature | Adjust the temperature of LLM within the range of 0 to 2, higher temperature mean more creativity, Default is 0.1         |   float   | Option |
| - e        | --extract     | Extract human voice from audio or not (Mac with apple silicon is not supported), Default is False |  Boolean  | Option |
| - b        | --batch       | Use 'True' if the input text file includes multiple file paths, Default is False             | 	Boolean | Option |
| - v        | --vectorDB    | Choose the vector database (pinecoene or chroma), Default is None                            | 	string  | Option |
| - m        | --model       | Choose the whisper model ('tiny', 'base', 'small', 'medium', 'large-v2'), Default is medium  |	  string  | Option |


## 🚀 Getting Started

***Dependencies***

Please ensure you have the following dependencies installed on your system:

`- Aanaconda or Miniconda`

`- python >=3.7, <=3.9 (Apple silicon python >= 3.8, <=3.9)`

`- pytorch`

### 🔧 Installation

1. Clone the Report.ai repository:
```sh
git clone https://github.com/Shou-Hsu/Report.ai.git
```

2. Change to the project directory:
```sh
cd Report.ai
```

3. Install the conda:
```sh
install minicode via https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html
```

4. Create virtual enviroment:
```sh
conda create -n Report.ai python=3.9
```

5. Activate virtual enviroment:
```sh
conda activate Report.ai
```

6. Install the pytorch:
```sh
install pytorch via https://pytorch.org/get-started/locally/
```

7. Install the ffmpeg and libsndfile:
```sh
conda install -c conda-forge ffmpeg libsndfile
```

8. Install the dependencies:
```sh
pip install -r requirements.txt
```


### 🤖 Running Report.ai

```sh
python main.py <file_path> -c 10000
```

### 🧪 Quickstart
1. Setting Openai or Azure openai crediential within the .env file. Furthermore, setting the credentials of either Pinecone or Chroma if aiming to store data in VectorDB.
```sh
# chioce one of gpt model provider Azure or OpenAI

# Azure openAI credential
AZURE_OPENAI_API_KEY=
AZURE_OPENAI_API_BASE=
AZURE_OPENAI_API_VERSION=
AZURE_OPENAI_API_TYPE=
AZURE_DEPLOYMENT_NAME=
EMBEDDING_DEPLOYMENT_NAME=  #only if you use Azure OpenAI

# # OpenAI credential
OPENAI_API_KEY=
MODEL_NAME=

# # pinecone credential (option)
PINECONE_API_KEY=
PINECONE_ENV=

# ChromaDB (option)
PERSIST_DIR=
COLLCTION_NAME=
```
2. Modify the tempelete/general.txt and tempelete/individuel.txt (Analysis items which seperated by ",")
```sh
#For instance, if you're aiming to analyze an "earnings call":
    you can set "Topic, Summary, CFO's explanation about short-term financial situation, CEO's description about the company's outlook, The issues of market concern" in tempelete/general.txt
    Simultaneously, set "Abstract, Investment insight, Keywords" in tempelete/individuel.txt

#In case you're looking to create a brief summary of the "routine meeting":
    you can set "Topic, Summary, Feature work" in tempelete/general.txt
    Simultaneously, set "Abstract, action item, Keywords" in tempelete/individuel.txt
```

3. Run Report.ai in commend line
```sh
python main.py example/WATCH_LIVE_Nvidia_Q2_Earnings_Call_NVDA.txt -c 10000 
```

---


## 🛣 Project Roadmap

> - [ ] `ℹ️  Publish project as a Python library via PyPI for easy installation.`
> - [ ] `ℹ️  Make project available as a Docker image on Docker Hub.`


---

## 🤝 Contributing

[**Discussions**](https://github.com/Shou-Hsu/Report.ai/discussions)
  - Join the discussion here.

[**New Issue**](https://github.com/Shou-Hsu/Report.ai/issues)
  - Report a bug or request a feature here.

[**Contributing Guidelines**](https://github.com/Shou-Hsu/Report.ai/blob/main/CONTRIBUTING.md)

## 📄 License


[MIT](https://choosealicense.com/licenses/).

---

## 👏 Acknowledgments

- Langchain, OpenAI, Pinecone, Chroma, Spleeter

[**Return**](#Top)

---

