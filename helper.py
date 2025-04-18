
from urlextract import URLExtract
extract = URLExtract()
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd
from collections import Counter
from emoji import is_emoji
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity




def fetch_stats(selected_user,df):

    if selected_user != 'Overall Analysis':
        df = df[df['sender'] == selected_user]

    # fetch the number of messages
    num_messages = df.shape[0]

    # fetch the total number of words
    words = []
    for message in df['message']:
        words.extend(message.split())

    # fetch number of media messages
    num_media_messages = df[df['message'].str.contains('image omitted', case=False, na=False)].shape[0]

    # fetch number of links shared
    links = []
    for message in df['message']:
        links.extend(extract.find_urls(message))

    return num_messages,len(words),num_media_messages,len(links)

def fetch_most_busy_users(df):
    x=df['sender'].value_counts().head()
    df=round(df['sender'].value_counts()/df.shape[0]*100,2).reset_index().rename(columns={'index':'name','sender':'percent'})
    name=x.index
    count=x.values
    return name,count,df


def create_wordcloud(selected_user,df):
    f = open('stop_hinglish.txt','r')
    stop_words = f.read()


    if selected_user != 'Overall Analysis':
        df = df[df['sender'] == selected_user]

    temp = df[df['sender'] != 'CS2 -grp B']
    temp = temp[~temp['message'].str.contains('image omitted', case=False, na=False)]

    def remove_stop_words(message):
        y=[]
        for word in message.lower().split():
            if word not in stop_words:
                y.append(word)
        return " ".join(y)

    wc = WordCloud(width=500,height=500,min_font_size=10,background_color='white')
    temp['message']=temp['message'].apply(remove_stop_words)
    df_wc = wc.generate(df['message'].str.cat(sep=" "))
    return df_wc


def most_common_words(selected_user,df):
    f = open('stop_hinglish.txt','r')
    stop_words = f.read()

    if selected_user != 'Overall Analysis':
        df = df[df['sender'] == selected_user]

    temp = df[df['sender'] != 'CS2 -grp B']
    temp = temp[~temp['message'].str.contains('image omitted', case=False, na=False)]

    words = []

    for message in temp['message']:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)

    most_common_df = pd.DataFrame(Counter(words).most_common(20))
    return most_common_df

def emoji_helper1(selected_user,df):
    if selected_user != 'Overall Analysis':
        df = df[df['sender'] == selected_user]
    
    emojis = []
    for message in df['message']:
        emojis.extend([c for c in message if is_emoji(c)])
    
    emoji_df=pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))

    return emoji_df

def monthly_timeline(selected_user,df):

    if selected_user != 'Overall Analysis':
        df = df[df['sender'] == selected_user]
    

    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index() 

    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + "-" + str(timeline['year'][i]))

    timeline['time'] = time
    return timeline

    
def daily_timeline(selected_user,df):
     if selected_user != 'Overall Analysis':
        df = df[df['sender'] == selected_user]

     daily_timeline = df.groupby('only_date').count()['message'].reset_index()

     return daily_timeline

def week_activity_map(selected_user,df):  
    if selected_user != 'Overall Analysis':
        df = df[df['sender'] == selected_user]

    return df['day_name'].value_counts()   
     
def month_activity_map(selected_user,df):  
    if selected_user != 'Overall Analysis':
        df = df[df['sender'] == selected_user]

    return df['month'].value_counts() 

def activity_heatmap(selected_user,df):
    if selected_user != 'Overall Analysis':
        df = df[df['sender'] == selected_user]

    
    user_heatmap=df.pivot_table(index='day_name',columns='period',values='message',aggfunc='count').fillna(0)

    return user_heatmap


# Function to perform sentiment analysis on messages
def get_sentiment(message):
    blob = TextBlob(message)
    sentiment_score = blob.sentiment.polarity  # Sentiment polarity between -1 (negative) and 1 (positive)
    
    if sentiment_score > 0:
        return 'Positive'
    elif sentiment_score == 0:
        return 'Neutral'
    else:
        return 'Negative'
    
# Modify the fetch_stats function to include sentiment analysis for each user
def fetch_user_sentiment(selected_user, df):
    if selected_user != 'Overall Analysis':
        df = df[df['sender'] == selected_user]
    
    sentiments = df['message'].apply(get_sentiment)
    sentiment_counts = sentiments.value_counts()

    return sentiment_counts

def sentiment_leaderboard(df):
    
    users = df['sender'].unique()
    users = [user for user in users if user != 'CS2 -grp B']  # Remove group/system messages

    user_sentiments = []

    for user in users:
        user_df = df[df['sender'] == user]
        polarities = user_df['message'].apply(lambda msg: TextBlob(msg).sentiment.polarity)
        avg_sentiment = polarities.mean()
        user_sentiments.append((user, avg_sentiment))

    sentiment_df = pd.DataFrame(user_sentiments, columns=['User', 'Avg Sentiment']).sort_values(by='Avg Sentiment')

    return sentiment_df


def compute_similarity(user1, user2, df):
    # Filter messages of both users
    user1_messages = df[df['sender'] == user1]['message']
    user2_messages = df[df['sender'] == user2]['message']

    # Combine the messages of both users into a list
    combined_messages = list(user1_messages) + list(user2_messages)

    # Vectorize the messages using TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(combined_messages)

    # Compute cosine similarity between user1's and user2's messages
    similarity_score = cosine_similarity(tfidf_matrix[0:len(user1_messages)], tfidf_matrix[len(user1_messages):])

    # Return the average similarity score
    return similarity_score.mean()




def generate_similarity_analysis(selected_user, df, user_list):
    user_similarity = {}
    
    # Calculate similarity between all users or for a selected user
    if selected_user == 'Overall Analysis':
        valid_users = [user for user in user_list if user != "Overall Analysis"]
        
        for user1 in valid_users:
            user_similarity[user1] = []
            for user2 in valid_users:
                if user1 == user2:
                    user_similarity[user1].append(1.0)
                else:
                    score = compute_similarity(user1, user2, df)
                    user_similarity[user1].append(score)

        similarity_df = pd.DataFrame(user_similarity, index=valid_users)
        return similarity_df

    else:
        other_users = [user for user in user_list if user not in ["Overall Analysis", selected_user]]
        similarity_scores = []

        for user in other_users:
            score = compute_similarity(selected_user, user, df)
            similarity_scores.append((user, score))

        result_df = pd.DataFrame(similarity_scores, columns=["User", "Similarity (Cosine)"])
        return result_df
