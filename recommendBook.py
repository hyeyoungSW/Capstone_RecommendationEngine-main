import csv
from os import error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import json


def setDescriptionSimilarity(description):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(description)
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    return cosine_sim

class Book:
    def __init__(self):
        self.content_df = pd.read_csv("final_book_upload_data.csv")
        self.review_df = pd.read_csv("book_review_data.csv")
        self.description_similarity = setDescriptionSimilarity(self.content_df['description'])
        self.feature = ['idx', 'title', 'etc', 'description', 'url']


    def recommendByUserEmotion(self, init_emotion, goal_emotion, idx_list):
        try:
            user_init = pd.DataFrame([init_emotion])
            user_goal = pd.DataFrame([goal_emotion])

            # find the userinitmood -> usergoalmood vector
            #user_vector = user_goal - user_init
            user_vector = user_goal
            # find the userinitmood -> reviewmood vector
            user_vector_list = user_vector.values.tolist()

            review_emotionDF = pd.DataFrame(data=self.review_df, columns=['review_sadness', 'review_joy', 'review_fear', 'review_disgust', 'review_anger'])
            # review_vector = pd.DataFrame(user_vector_list*(int)(review_emotionDF.size/5), columns=['review_sadness', 'review_joy', 'review_fear', 'review_disgust', 'review_anger'])
            # review_vector = review_emotionDF - review_vector
            review_vector = review_emotionDF
            cosine_sim = linear_kernel(user_vector, review_vector)
            sim_scores = list(enumerate(cosine_sim[0]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            #book_indices = [i[0] for i in sim_scores[0:2]]         
               
            is_hated = {}
            for i in idx_list:
                is_hated[int(i)] = True

            book_indices = []

            for i in sim_scores: 
                book = self.content_df.iloc[i[0]]
                book_idx = book['idx']
                if(int(book_idx) in is_hated): 
                    continue
                book_indices.append(i[0])
                if(len(book_indices) == 4):
                    break

            closest_items = self.review_df.iloc[book_indices]

            return closest_items[self.feature]
        except:
            return error

    def recommendByUserSentence(self, goal_sentence, idx_list):
        try:
            vector_new = self.content_df['description'].copy()
            vector_new.loc[len(vector_new)] = goal_sentence
            tfidf = TfidfVectorizer(stop_words='english')
            tfidf_matrix = tfidf.fit_transform(vector_new)

            cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
            idx = vector_new.size-1
            sim_scores = list(enumerate(cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

            is_hated = {}
            for i in idx_list:
                is_hated[int(i)] = True

            book_indices = []

            for i in sim_scores[1:]: 
                book = self.content_df.iloc[i[0]]
                book_idx = book['idx']
                if(int(book_idx) in is_hated): 
                    continue
                book_indices.append(i[0])
                if(len(book_indices) == 2):
                    break

            closest_items = self.content_df.iloc[book_indices]
            return closest_items[self.feature]
        except:
            return error

    def recommendByDescription(self, title, top):
        indices = pd.Series(self.content_df.index, index=self.content_df['title'])
        idx = indices[title]
        sim_scores = list(enumerate(self.description_similarity[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:1+int(top)]
        movie_indices = [i[0] for i in sim_scores]
        result = self.content_df.iloc[movie_indices]

        return result[self.feature]

    def recommendByItemContent(self, items, top):
        try:
            result = pd.DataFrame(columns = self.feature)

            for title in items:
                # recommend by description
                recommended_list = self.recommendByDescription(title, top)
                result = pd.concat([result, recommended_list[self.feature]], ignore_index=True)

            result = result.drop_duplicates(subset=['idx'])
            return {"type" : "book", "total_count" : len(result), "items" : result.to_json(orient='records')}
        except:
            return error

    def getItemByIndex(self, idx_list):
        try:
            result = pd.DataFrame(columns = self.feature)

            for i in idx_list:
                idx = int(i)
                item = self.content_df.loc[self.content_df['idx'] == idx]
                result = pd.concat([result, item[self.feature]], ignore_index=True)
            return result.to_json(orient='records')
        except:
            return error