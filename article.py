from transformers import pipeline

news = pipeline('text-generation', model='./gpt2-news', tokenizer="dbmdz/german-gpt2")

artikel = news('Das britische Parlament soll am Freitag zum dritten Mal über den Brexit-Deal mit der Europäischen Union abstimmen. Das teilte die für Parlamentsfragen zuständige Ministerin Andrea Leadsom im Unterhaus mit.', max_length=230)[0]['generated_text']

print(artikel)
