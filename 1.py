from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer,AlbertForSequenceClassification,AutoModelForSequenceClassification,RobertaTokenizer,BertTokenizer

# # Sentiment analysis pipeline
# analyzer = pipeline("sentiment-analysis")

# # Question answering pipeline, specifying the checkpoint identifier
# oracle = pipeline(
#     "question-answering", model="distilbert-base-cased-distilled-squad", tokenizer="bert-base-cased"
# )
aaaa='Jiabo/Roberta_Chinese_sentiment'
# Named entity recognition pipeline, passing in a specific model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained(aaaa)
tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
#参考这个配置https://huggingface.co/Jiabo/Roberta_Chinese_sentiment/blob/main/config.json
recognizer = pipeline("text-classification", model=model, tokenizer=tokenizer)


print(recognizer('我草'))
print(recognizer('我喜欢你'))