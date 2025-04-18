import streamlit as st
import pandas as pd
import numpy as np
import preprocess
import helper
import matplotlib.pyplot as plt
import seaborn as sns



st.sidebar.title("Whatsapp Chat Analyzer")

uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    df = preprocess.preprocess(data)

    st.write(df)

    # fetch unique users
    user_list = df['sender'].unique().tolist()
    user_list.remove('CS2 -grp B')
    user_list.sort()
    user_list.insert(0, "Overall Analysis")



    selected_user = st.sidebar.selectbox("Select a user", user_list)

    

    if st.sidebar.button("Show Analysis"):
        
        #stats area
        num_messages, words, num_media_messages,num_links =helper.fetch_stats(selected_user,df)

        col1 , col2, col3, col4 = st.columns(4)


        with col1:
            st.header("Total Messages")
            st.title(num_messages)
            

        with col2:
            st.header("Total Words")
            st.title(words)
        
        with col3:
            st.header("Media Shared")
            st.title(num_media_messages)

        with col4:
            st.header("Links Shared")
            st.title(num_links)


        
        # monthly timeline
        st.title("Monthly Timeline")

        timeline = helper.monthly_timeline(selected_user,df) 
         
        fig,ax = plt.subplots()
        ax.plot(timeline['time'],timeline['message'],color='green')
        xticks = plt.xticks(rotation=90)
        st.pyplot(fig)


        # daily timeline
        st.title("Daily Timeline")

        daily_timeline = helper.daily_timeline(selected_user,df) 
         
        fig,ax = plt.subplots()
        ax.plot(daily_timeline['only_date'],daily_timeline['message'],color='black')
        xticks = plt.xticks(rotation=90)
        st.pyplot(fig) 


        # activity map
        st.title("Activity Map")

        col1,col2 = st.columns(2)

        with col1:
            st.header("Most Busy Day")
            busy_day = helper.week_activity_map(selected_user,df)
            fig,ax = plt.subplots()
            ax.bar(busy_day.index,busy_day.values)
            st.pyplot(fig)

        with col2:
            st.header("Most Busy Month")
            busy_month = helper.month_activity_map(selected_user,df)
            fig,ax = plt.subplots()
            ax.barh(busy_month.index,busy_month.values)
            xticks = plt.xticks(rotation=90)
            st.pyplot(fig)

        user_heatmap=helper.activity_heatmap(selected_user,df)
        fig,ax = plt.subplots()
        ax=sns.heatmap(user_heatmap)
        st.pyplot(fig)




        # finding busiest users in group(Group level)
        
        if selected_user == 'Overall Analysis':
            st.title("Busiest Users")

            name,count,new_df=helper.fetch_most_busy_users(df)
            fig, ax = plt.subplots()
            

            col1, col2 = st.columns(2)

            with col1:
                 ax.bar(name,count)
                 plt.xticks(rotation='vertical')
                 st.pyplot(fig)

            with col2:
                st.dataframe(new_df)

        #WordCloud
        st.title("Word Cloud")
        df_wc=helper.create_wordcloud(selected_user,df)
        fig,ax = plt.subplots() 
        ax.imshow(df_wc)
        st.pyplot(fig)


        #Most Common Words
        most_common_df=helper.most_common_words(selected_user,df)

        fig,ax = plt.subplots()

        ax.barh(most_common_df[0],most_common_df[1])
        plt.xticks(rotation='vertical')

        st.pyplot(fig)

        st.title("Most Common Words")
        st.dataframe(most_common_df)

        
        #Emoji Analysis
        

        emoji_df=helper.emoji_helper1(selected_user,df)
        st.title("Emoji Analysis")

        col1,col2 = st.columns(2)

        with col1:
            st.dataframe(emoji_df) 
        
        with col2:
            fig,ax = plt.subplots()
            ax.pie(emoji_df[1].head(),labels=emoji_df[0].head(),autopct="%0.2f")
            st.pyplot(fig)


        #Sentiment Analysis
        st.title("Sentiment Analysis")
        sentiment_counts = helper.fetch_user_sentiment(selected_user, df)
        st.title(f"Sentiment Analysis for {selected_user}")

        # Display sentiment counts
        st.write(sentiment_counts)

        #Sentiment Leaderboard
        st.title("Who Was the Most Positive or Negative?")

        sentiment_df = helper.sentiment_leaderboard(df)

        # Bar plot for average sentiment per user
        fig, ax = plt.subplots(figsize=(12, 8))
        
        colors = sentiment_df['Avg Sentiment'].apply(lambda x: 'green' if x > 0 else 'red')
        ax.barh(sentiment_df['User'], sentiment_df['Avg Sentiment'], color=colors)
        ax.set_xlabel('Average Sentiment')
        ax.set_ylabel('Users')
        ax.set_title('Average Sentiment per User')
        st.pyplot(fig)

        # Highlight most positive and most negative
        most_positive = sentiment_df.iloc[-1]
        most_negative = sentiment_df.iloc[0]

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Most Positive User", most_positive['User'], f"{most_positive['Avg Sentiment']:.2f} 😊")
        with col2:
            st.metric("Most Negative User", most_negative['User'], f"{most_negative['Avg Sentiment']:.2f} 😠")


        # dislay the similarity

        similarity_df = helper.generate_similarity_analysis(selected_user, df, user_list)

        if selected_user == 'Overall Analysis':
            # Show Similarity Matrix for Overall Analysis
            st.subheader("Overall Pairwise Similarity Matrix")
            st.dataframe(similarity_df.round(2))

            # Plotting the Heatmap
            st.subheader("Similarity Heatmap")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(similarity_df, annot=True, cmap="YlGnBu", fmt=".2f")
            st.pyplot(fig)

        else:
            # Show Similarity Scores for Selected User
            st.subheader(f"Similarity of {selected_user} with Other Users")
            st.dataframe(similarity_df.sort_values(by="Similarity (Cosine)", ascending=False).reset_index(drop=True))

            # Plotting the bar chart for similarity scores
            st.subheader("Visual Comparison")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(similarity_df["User"], similarity_df["Similarity (Cosine)"], color='skyblue')
            ax.set_xlabel("Cosine Similarity")
            st.pyplot(fig)




