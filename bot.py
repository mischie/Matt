import json
import random
import nltk
import pickle
import numpy as np
import discord

from discord.ext import commands
from nltk import WordNetLemmatizer
from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()

loaded_model = load_model("chatbot_model.h5")
loaded_intents = json.loads(open("intents.json").read())
loaded_words = pickle.load(open("words.pkl", "rb"))
loaded_classes = pickle.load(open("classes.pkl", "rb"))


def clean_up_sentence(sentence) -> list:
    """Clean up the given sentence by lemmatization"""
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence


def bow(sentence, words, show_details=True) -> np.array:
    """Bag the matching sentence"""
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)


def predict_class(sentence, model) -> list:
    """Predict the sentence"""
    # filter out predictions below a threshold
    p = bow(sentence, loaded_words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": loaded_classes[r[0]], "probability": str(r[1])})
    return return_list


def getResponse(ints, intents_json) -> str:
    """Get the response"""
    global result
    tag = ints[0]["intent"]
    list_of_intents = intents_json["intents"]
    for i in list_of_intents:
        if i["tag"] == tag:
            result = random.choice(i["responses"])
            break
    return result


def get_chatbot_response(msg) -> str:
    """Get respose from the matched sentence"""
    ints = predict_class(msg, loaded_model)
    res = getResponse(ints, loaded_intents)
    return res


def load_prefix():
    return commands.when_mentioned_or("c.", "c?", "c? ", "c. ")


bot = commands.Bot(command_prefix=load_prefix())


@bot.event
async def on_ready():
    print("Bot started")


@bot.command()
async def talk(ctx, *, query):
    """Talk to the chatbot!"""
    await ctx.trigger_typing()
    res = get_chatbot_response(query)
    await ctx.send(f"> {query}\n\nChatbot: {res}")


if __name__ == "__main__":
    bot.run("replace with your token")
