from flask import Flask, render_template, request
import recommendBook
import recommendMovie
import csv
import json

movieContent = recommendMovie.Movie()
bookContent = recommendBook.Book()

app = Flask(__name__)

@app.route('/')
def main():
    return {'testpage':'hello world'}

@app.route('/movie/sentence', methods=['POST'])
def getMovieBySentence():
    user_status = request.form 
    goal_sentence = user_status['sentence']
    idx_list = json.loads(request.form.to_dict()['idx_list'])
    #language = user_status["language"]
    recommendBySentence = movieContent.recommendByUserSentence(goal_sentence, idx_list)
    
    return recommendBySentence.to_json(orient="records")

@app.route('/movie/emotion', methods=['POST'])
def getMovieByEmotion():
    user_status = request.form
    init_emotion = json.loads(user_status['init_emotion'])
    goal_emotion = json.loads(user_status['goal_emotion'])
    idx_list = json.loads(request.form.to_dict()['idx_list'])
    recommendByEmotion = movieContent.recommendByUserEmotion(init_emotion, goal_emotion, idx_list)
    
    return recommendByEmotion.to_json(orient="records")

@app.route('/movie/content', methods=['POST'])
def getMovieByItemContent():
    user_status = request.form.to_dict()
    item_list = json.loads(user_status["selected_items"])
    counts_per_item = user_status["counts_per_item"]
    recommendedMovieLists = movieContent.recommendByItemContent(item_list, counts_per_item)

    return json.dumps(recommendedMovieLists)

@app.route('/movie', methods=['POST'])
def getMovieByIndex():
    user_status = request.form.to_dict()
    idx_list = json.loads(user_status["idx_list"])
    #language = json.loads(user_status["language"])

    item_list = movieContent.getItemByIndex(idx_list)

    return json.dumps(item_list)

@app.route('/book/sentence', methods=['POST'])
def getBookBySentence():
    user_status = request.form 
    goal_sentence = user_status['sentence']
    idx_list = json.loads(request.form.to_dict()['idx_list'])
    recommendBySentence = bookContent.recommendByUserSentence(goal_sentence, idx_list)

    return recommendBySentence.to_json(orient="records")

@app.route('/book/emotion', methods=['POST'])
def getBookByEmotion():
    user_status = request.form
    init_emotion = json.loads(user_status['init_emotion'])
    goal_emotion = json.loads(user_status['goal_emotion']) 
    idx_list = json.loads(request.form.to_dict()['idx_list'])
    recommendByEmotion = bookContent.recommendByUserEmotion(init_emotion, goal_emotion, idx_list)
    
    return recommendByEmotion.to_json(orient="records")

@app.route('/book/content', methods=['POST'])
def getBookByItemContent():
    user_status = request.form.to_dict()
    item_list = json.loads(user_status["selected_items"])
    counts_per_item = user_status["counts_per_item"]
    recommendedBookLists = bookContent.recommendByItemContent(item_list, counts_per_item)

    return json.dumps(recommendedBookLists)

@app.route('/book', methods=['POST'])
def getBookByIndex():
    user_status = request.form.to_dict()
    idx_list = json.loads(user_status["idx_list"])
    item_list = bookContent.getItemByIndex(idx_list)

    return json.dumps(item_list)

if __name__ == '__main__':
    app.run()