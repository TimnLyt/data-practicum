import nltk
nltk.download('punkt')
nltk.download('wordnet')
from flask import Flask, render_template, request
from flask_ngrok import run_with_ngrok
import numpy as np
import pickle
import json
import random
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
import pandas as pd
from collections import Counter


app = Flask(__name__)
# run_with_ngrok(app)

@app.route("/")
def home():
    return render_template("index.html")

imdb = pd.read_csv("imdb_top_1000.csv")
summary = []
overview = imdb[imdb.columns[7]]
movie_list = imdb[imdb.columns[1]]
lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))
model = load_model('chatbot_model.h5')

genre_list_db = imdb.Genre.unique()
list_set = set()
for i in range(len(genre_list_db)):
  text_1 = genre_list_db[i].split(',')
  for j in range(len(text_1)):
    list_set.add(text_1[j].replace(" ", ""))
unique_genre_list = (list(list_set))
unique_genre_list = (map(lambda x: x.lower(), unique_genre_list))
unique_genre_list = list(unique_genre_list)


def get_emotion(summary):
  emotion_list = []
  with open('emotions.txt', 'r') as file:
    for line in file:
        clear_line = line.replace("\n", '').replace(",", '').replace("'", '').strip()
        word, emotion = clear_line.split(':')

        if word in summary:
            emotion_list.append(emotion)

  w = Counter(emotion_list)
  if len(w) > 0:
    emotion = max(w, key = w.get)
    return emotion
  else:
    return "NaN"

for i in range(1000):
  summary.append(overview[i])

happy = []
sad = []

for h in range(1000):
  emotion = get_emotion(summary[h])
  if emotion == " happy":
    happy.append(h)
  elif emotion == " sad":
    sad.append(h)

print("happy = ", len(happy))
print("sad = ", len(sad))

def get_happy():
  happy_list = []
  for index in range(len(happy)):
    happy_list.append(movie_list[happy[index]])
  random.shuffle(happy_list)
  return happy_list

def get_sad():
  sad_list = []
  for index in range(len(sad)):
    sad_list.append(movie_list[sad[index]])
  random.shuffle(sad_list)
  return sad_list

def clean_up_sentence(sentence):
  sentence_words = nltk.word_tokenize(sentence)
  sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]

  return sentence_words

def bag_of_words(sentence):
  sentence_words = clean_up_sentence(sentence)
  bag = [0] * len(words)
  for w in sentence_words:
    for i, word in enumerate(words):
      if word == w:
        bag[i] = 1
  return np.array(bag)

def predict_class(sentence):
  bow = bag_of_words(sentence)
  res = model.predict(np.array([bow]))[0]
  ERROR_THRESHOLD = 0.25
  results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

  results.sort(key = lambda x: x[1], reverse = True)
  return_list = []
  for r in results:
    return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
  return return_list

def get_response(intents_list, intents_json):
  tag = intents_list[0]['intent']
  list_of_intents = intents_json['intents']
  for i in list_of_intents:
    if i['tag'] == tag:
      result = random.choice(i['responses'])
      break
  return result

def check_if_exists(sentence):
  movies_list = imdb[imdb.columns[1]].values
  for movie in movies_list:
    if movie in sentence:
      return movie
  return ''

def get_overview(movie):
  df = imdb.loc[imdb['Series_Title'] == movie]
  overview_temp = df['Overview'].values
  return overview_temp

def choose_movie(ints):
  movie_list = []
  sentence = ''
  movie_list.append("Which one do you like to watch?")
  if ints == "feel_sad":
    temp_list = get_happy()
    sentence = 'Hope this list of happy movies can make your day. '
  elif ints == "feel_happy":
    temp_list = get_sad()
    sentence = "Here are some sad movies, maybe you will have fun with them. "

  for i in range(5):
    movie_list.append(temp_list[i])

  movie_list = ','.join(movie_list)
  movie_list = sentence + movie_list

  return movie_list

def queue_movies(genre):

  df = imdb['Genre'].values

  queue_movie = []
  count = []
  num = 0

  for index in range(len(df)):
    temp = df[index].lower()
    if genre in temp:
      count.append(index)

  movie_list = imdb['Series_Title'].values
  random.shuffle(count)

  for i in count:
    if num > 9:
      return queue_movie
    queue_movie.append(movie_list[i])
    num = num + 1

  return queue_movie

def get_random_movies():

  movie_temp_list = []
  num = 0
  movies_list = imdb[imdb.columns[1]].values
  a = list(range(0, 1000))
  random.shuffle(a)

  for i in a:
    if num > 9:
      return movie_temp_list
    movie_temp_list.append(movies_list[i])
    num = num + 1

  return movie_temp_list

@app.route("/get")
def get_bot_response():
  message = request.args.get('msg')
  ints = predict_class(message)
  res = get_response(ints, intents)
  movie_select = ''

  if ints[0]['intent'] == "feel_happy" or ints[0]['intent'] == "feel_sad":
    movie_select = choose_movie(ints[0]['intent'])
    res = movie_select

  if ints[0]['intent'] == "Yes" or "Yes" in message:
    res = check_if_exists(message)
    if len(res) > 0:
      res = "The " + res + "will start soon."
    else:
      res = "Sorry, I am not sure what do you meant"

  if ints[0]['intent'] == "genres":
    user_genre = ''
    for genre in unique_genre_list:
      if genre in message:
        user_genre = genre
        res = queue_movies(user_genre)
        res = ','.join(res)
        res = "which one do you like to watch? " + res
    if len(res) < 1:
      res = "Sorry, we don't have that type of movies"

  if ints[0]['intent'] == "overview":
    movie_name = check_if_exists(message)
    if len(movie_name) < 1:
      res = "Sorry, we don't have this movies"
    else:
      movie_overview = get_overview(movie_name)
      res = " ".join(movie_overview)

  if ints[0]['intent'] == "movies":
    move = get_random_movies()
    res = ",".join(move)
    res = "Here is some popular movies, " + res

  return res

if __name__ == "__main__":
    app.run()(debug=True, host='0.0.0.0')