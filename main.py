from utils import credential_validation, get_file_list, validation_and_filetype_check, detect_language
from storage_vector import pinecone_storage, chroma_storage
from s2t_whisper import speech2text, download_from_youtube
from summarize import generate_summary
from divide import divide_article
import argparse, os

def main():
    parser = argparse.ArgumentParser(description='Build your own professional database')
    parser.add_argument('file_path', type=str, help='file path')
    parser.add_argument('-c', '--chunk', default=2000, type=int, help='chunk size')
    parser.add_argument('-t', '--temperature', default=0.1, type=float, help='temperature of LLM')
    parser.add_argument('-b', '--batch', default=False, action="store_true", help='batch process')
    parser.add_argument('-l', '--translated_language', default='zh-tw', help='the language that should be translated')
    parser.add_argument('-e', '--extract', default=False, action="store_true", help='Extract human voice from audio (not support in Apple silicon)')
    parser.add_argument('-o', '--output_dir', default='./docx', type=str, help='file path of output report')
    parser.add_argument('-v', '--vectorDB', default=None, choices=['pinecone', 'chroma', None], help='select the vectorDB')
    parser.add_argument('-m', '--model', type=str, default='medium', help='the using model for ASR',
                        choices=['tiny', 'base', 'small', 'medium', 'large-v2'])

    args = parser.parse_args()

    # credential validation
    credential_validation(vectorDB=args.vectorDB, temperature=args.temperature)

    if args.batch: 
        file_list = get_file_list(file_path=args.file_path)
    else: 
        file_list = [args.file_path]

    for file_path in file_list:
        # validate the type of input file 
        file_type, file_name = validation_and_filetype_check(file_path, args.output_dir)
        print(f'Strat analysis {file_name}')
        if file_type == 'url':
            file_name = download_from_youtube(file_path)
            language = speech2text(file_name=file_name, model_name=args.model, extraction=args.extract)
        elif file_type == 'audio':
            language = speech2text(file_name=file_name, model_name=args.model, extraction=args.extract)
        elif file_type == 'transcript':
            language = detect_language(file_path)

        # divide the article and generate summary
        article_divided = divide_article(file_name=file_name, original_language=language, chunk_size=args.chunk).run()
        generate_summary(file_name=file_name, original_language=language, 
            translated_language=args.translated_language, chunk_size=args.chunk, output_dir=args.output_dir).run(article_divided)

        # Pinecone only provide one index for free account
        if args.vectorDB == 'pinecone':
            pinecone_storage(file_name=file_name)
        elif args.vectorDB == 'chroma':
            chroma_storage(file_name=file_name, collection_name=os.getenv('COLLCTION_NAME'),persist_directory=os.getenv('PERSIST_DIR'))

if __name__ == "__main__":
    main()
