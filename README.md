# Analyzing Car Reviews with LLMs

## Project Overview
**Car-ing is sharing**, an auto dealership, wants to implement a chatbot to assist customers and human agents. This prototype leverages a variety of pre-trained Large Language Models (LLMs) from Hugging Face to perform core NLP tasks on customer car reviews: Sentiment Analysis, Translation, and Extractive Question Answering.

## Dataset
- **Files**: `data/car_reviews.csv` (Reviews and true sentiment classes), `data/reference_translations.txt`

## Technologies Used
- **Python**: Pandas, PyTorch
- **NLP / Hugging Face**: `transformers` (`pipeline`, `AutoTokenizer`, `AutoModelForQuestionAnswering`), `evaluate`

## Methodology
1. **Sentiment Classification**: 
   - Utilized `distilbert-base-uncased-finetuned-sst-2-english` via a pipeline to classify reviews as POSITIVE or NEGATIVE.
   - Evaluated the predictions against real labels using Accuracy and F1 score metrics.
2. **Text Translation**: 
   - Used `Helsinki-NLP/opus-mt-en-es` to translate English car reviews into Spanish.
   - Assessed translation quality against reference text using the BLEU score.
3. **Extractive QA**: 
   - Tokenized the text and ran `deepset/minilm-uncased-squad2` to answer specific questions based on the review's context (e.g., "What did he like about the brand?").
   - Extracted answers using PyTorch `argmax` over the start and end logits.
