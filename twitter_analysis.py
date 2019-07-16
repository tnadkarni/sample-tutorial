#import all necessary packages for collecting/manipulating/visualizing data
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

tweets_data = []

#for git testing

tweets_data_path = 'twitterdata.txt' #specify name of twitter file

#convert JSON data format into elements of Python list
tweets_file = open(tweets_data_path, "r")
for line in tweets_file:
    try:
        tweet = json.loads(line)
        tweets_data.append(tweet)
    except:
        continue

print(len(tweets_data)) #get number of tweets

tweets = pd.DataFrame() #create dataframe to organize data

#Some values in the file may be connection erorr messages instead of tweets, remove them
for i in range(len(tweets_data)):
    try:
        y = tweets_data[i]['id']
    except KeyError:
        tweets_data[i] = 0
tweetdat = [x for x in tweets_data if x!=0]

#Add observations to dataframe
tweets['text'] = list(map(lambda tweet: tweet['text'], tweetdat))
tweets['location'] = list(map(lambda tweet: tweet['user']['location'], tweetdat)) 
tweets['followers'] = list(map(lambda tweet: tweet['user']['followers_count'], tweetdat)) 
tweets['lang'] = list(map(lambda tweet: tweet['user']['lang'], tweetdat)) 
tweets['time'] = list(map(lambda tweet: tweet['created_at'], tweetdat))

##Set up a python dict to be able to capture when a tweet is talking about a particular commerical
ads = {'Amazon':['amazon', 'alexa', 'rebel wilson', 'anothony hopkins', 'jeff bezos', 'cardi b'], 
       'AFM':['avocado', 'mexico', 'guac'], 
       'Bud Light':['bud light', 'dilly'], 
       'Budweiser':['budweiser'], 
       'Coke':['coke','gillian jacobs'], 
       'Doritos':['doritos', 'dinklage', 'busta rhymes', 'spitfire'],
       'Mountain Dew':['dew', 'freeman','missy elliot'], 
       'E*Trade':['e trade', 'etrade'], 
       'febreze':['febreze', 'P&amp;G'], 
       'Groupon':['groupon', 'tiffany'], 
       'Hyundai':['hyundai', 'soldiers'], 
       'Intuit/TurboTax':['turbotax','intuit'],
       'Jack in the Box':['jackvsmartha', 'jack in the box', 'martha stewart'], 
       'Kia':['kia', 'aerosmith', 'steven tyler'], 
       'Lexus':['lexus', 'black panther'], 
       'M&Ms':['M&amp;M', 'devito'], 
       'Michelob Ultra':['Michelob', 'chris pratt'], 
       'Monster Products':['monster', 'iggy azalea'], 
       'Pepsi':['crawford', 'britney','pepsi'],
       'Persil':['persil', 'peter hermann'],
       'Pringles':['pringle', 'bill hader'],
       'Quicken Loans':['quicken', 'rocket', 'mortgage'],
       'Skittles':['skittles', 'david schwimmer'],
       'Sprint':['sprint', 'evelyn'],
       'Squarespace':['squarespace', 'keanu reaves'],
       'Stella Artois':['chalice', 'stella ', 'water.org', 'matt damon'],
       'Tide':['tide', 'david harbour'],
       'Toyota':['toyota'],
       'Turkish Airlines':['turkish', 'dr.oz'],
       'Universal Orlando':['peyton manning', 'harry potter', 'orlando', 'universal'],
       'Verizon':['verizon'],
       'WeatherTech':['weathertech'],
       'Wendys':['wendy', 'mcdonald']
        } 

import re
#Function to find presence of word in tweet
def words_in_text(l, text):
    if re.compile('|'.join(l), re.IGNORECASE).search(text):
        return 1
    return 0

mentions = {}
for x in ads.keys():
    tweets[x] = tweets['text'].apply(lambda tweet: words_in_text(ads[x], tweet))
    mentions[x]=sum(tweets[x])

#Get the top 10 talked about commercials
top_mentions = sorted(mentions, key=mentions.get, reverse=True)[0:9]
print(top_mentions)

#Plot the top mentions
x = []
y = []
for i in top_mentions:
    x.append(i)
    y.append(mentions[i])

x_pos = list(range(len(x)))
width = 0.5
fig, ax = plt.subplots()
plt.bar(x_pos, y, width, alpha=1, color='g')

# Setting axis labels and ticks
ax.set_ylabel('Number of tweets', fontsize=17)
ax.set_title('Ad Mentions', fontsize=17, fontweight='bold')
ax.set_xticks([p for p in x_pos])
ax.set_xticklabels(x, fontsize=12)
plt.grid()
plt.show()

#Get only Media related tweets i.e. talking about atleast one of the commericials

tweets2 = tweets.copy()
tweets2.drop(tweets2.iloc[:, 0:9], inplace=True, axis=1)
tweets['media'] = tweets2.sum(axis=1)
mediatweets = tweets[tweets['media']>0]

#Extract hashtags using regex
y = mediatweets['text'].str.findall(r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)")
tags = [item[1::] for sublist in y for item in sublist]

#Write to csv
import csv
f = open('tags.csv', 'w', encoding='utf-8-sig')
w = csv.writer(f, delimiter=',')
w.writerows([x.split(' ') for x in tags])
f.close()

import matplotlib as mpl
from wordcloud import WordCloud, STOPWORDS

#Add non-media related hashtags to stopwords since we're only interested in what was said about commercials 
stopwords = set(STOPWORDS) 
nonmedia = ['SuperBowl', 'eagles','philly', 'philadelphia', 'selfiekid', 'SB52', 'SuperBowlChamps', 
            'SBLII', 'Rihanna', 'Shakira', 'Pink', 'Jayz', 'freerangekids', 'mariahcarey',
           'superbowllII', 'flyeaglesfly', 'philadelphiaeagles', 'halftimeselfie', 'britneyspears']
stopwords.update(nonmedia)
            
data = pd.read_csv('tags.csv')

mpl.rcParams['font.size']=12                 
mpl.rcParams['savefig.dpi']=100          
mpl.rcParams['figure.subplot.bottom']=.1 

#Create the wordcloud
wordcloud = WordCloud(
                          background_color='white',
                          stopwords=stopwords,
                          max_words=200,
                          max_font_size=40, 
                          random_state=42
                         ).generate(str(data))
#Save the figure
fig = plt.figure(1)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
fig.savefig("word1.png", dpi=900)

#Basic sentiment analysis using Python library, you may train your own classifier 
from textblob import TextBlob

def clean_tweet(tweet):
    '''
    Utility function to clean the text in a tweet by removing 
    links and special characters using regex.
    '''
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())


def get_sentiment(tweet):
    analysis = TextBlob(clean_tweet(tweet))
    if analysis.sentiment.polarity>0:
        return 'Postive'
    elif analysis.sentiment.polarity==0:
        return 'Neutral'
    else:
        return 'Negative';

#Get count of tweets with each sentiment for each commercial
mediatweets['sentiment'] = mediatweets['normalized_text'].apply(get_sentiment)
sentiment = pd.DataFrame(columns=['Postive', 'Neutral', 'Negative'])
adlist = sorted(mentions, key=mentions.get, reverse=True)[0:13]

for ad in adlist:
    x = mediatweets.loc[mediatweets[ad]==1, 'sentiment']
    pos = x.value_counts().loc['Postive']
    neu = x.value_counts().loc['Neutral']
    neg = x.value_counts().loc['Negative']
    sentiment.loc[ad] = [pos, neu, neg]

#Plot a grouped barchart

barWidth = 0.2
 
# set height of bar
bars1 = sentiment['Postive']
bars2 = sentiment['Neutral']
bars3 = sentiment['Negative']
 
# Set position of bar on X axis
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
 
# Make the plot
plt.bar(r1, bars1, color='#00cc00', width=barWidth, edgecolor='white', label='Postive')
plt.bar(r2, bars2, color='#0080ff', width=barWidth, edgecolor='white', label='Neutral')
plt.bar(r3, bars3, color='#ff0000', width=barWidth, edgecolor='white', label='Negative')
 
# Add xticks on the middle of the group bars
plt.xlabel('Advertisement', fontweight='bold', fontsize=20)
plt.ylabel('Number of Tweets', fontweight='bold', fontsize=20)
plt.xticks([r + barWidth for r in range(len(bars1))], list(sentiment.index.values), fontsize=17)
plt.yticks(fontsize=17)
 
# Create legend & Show graphic
plt.legend(fontsize=20)
plt.rcParams["figure.figsize"] = (30,10)
plt.show()

#Get some other stats
top_langs = mediatweets['lang'].value_counts()[0:4] #Top 5 languages spoken by twitter audience
graph(top_locs.index, top_locs.values, 'blue', 'Ad Tweet location origin', 'Ad tweet count')

top_locs = mediatweets['location'].value_counts()[0:4] #Top 5 registered locations of twitter users
graph(top_locs.index, top_locs.values, 'blue', 'Ad Tweet location origin', 'Ad tweet count')



