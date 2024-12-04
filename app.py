from flask import Flask, request, jsonify, render_template
import os
from apis import get_user_faces, judge_faces, set_others_faces, train_cnn, load_data

app = Flask(__name__)

USER_FACES_DIR = 'user_faces'
OTHER_FACES_DIR = 'others_faces'
if not os.path.exists(USER_FACES_DIR):
    os.makedirs(USER_FACES_DIR)
if not os.path.exists(OTHER_FACES_DIR):
    os.makedirs(OTHER_FACES_DIR)
    
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/get_user_faces_by_cap', methods=['GET'])
def get_user_faces_by_cap():
    if get_user_faces.get_user_faces() == 0:
        return jsonify({'error': 'Taking photos of user failed!'})
    return jsonify({"message": "User faces captured successfully."})

@app.route('/api/find_model', methods=['GET'])
def find_trainned_model():
    if train_cnn.find_model() == 0:
        return jsonify({'error':'You have no models trainned!'})
    return jsonify({"message":"You have models trainned."})

@app.route('/api/set_others_faces', methods=['GET'])
def set_others_face():
    if set_others_faces.set_others_faces() == 0:
        return jsonify({'error': 'Uploading others\' faces failed!'})
    return jsonify({"message": "Other faces loaded successfully."})

@app.route('/api/train_cnn', methods=['POST'])
def train_CNN():
    if train_cnn.trainCNN() == 0:
            return jsonify({"error": "Trainning CNN model has failed!"})
    return jsonify({"message": "Model training complete."})
    

@app.route('/api/judge_user_face_by_cap', methods=['GET'])
def judge_user_face_by_cap():
    result = judge_faces.judge_faces_camera()
    if result == 1:
        return jsonify({"message":"Stop recognition."})

if __name__ == '__main__':
    app.run(debug=True)