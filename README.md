AutoJudge

AutoJudge is a machine learning–based application that predicts the difficulty of programming problems using only their textual descriptions.
It classifies problems into Easy, Medium, or Hard and also outputs a numerical difficulty score for finer analysis.

Project Structure
Auto Judge/
│
├── data/
│   └── problems_data.jsonl
│
├── src/
│   ├── preprocess.py
│   ├── features.py
│   ├── train_classifier.py
│   ├── train_regressor.py
│   └── utils.py
│
├── models/
│   ├── vectorizer.pkl
│   ├── classifier.pkl
│   └── regressor.pkl
│
├── web/
│   ├── app.py
│   ├── templates/
│   │   └── index.html
│   └── static/
│       └── style.css
│
└── README.md



Overview

AutoJudge works by analyzing the text of a programming problem, including its description, input format, output format, and constraints.
It uses machine learning models trained on previously labeled problems to estimate how difficult a new problem is likely to be.

The system is intended for difficulty estimation, not for solving the problem itself.

How to Run the Project
1. Install Required Libraries
pip install -r requirements.txt

2. Generate Text Features
python src/features.py

3. Train Models
python src/train_classifier.py
python src/train_regressor.py

4. Start the Web Application
python web/app.py


Open your browser and go to:

http://127.0.0.1:5000

Output

For each problem statement entered, AutoJudge provides:

Predicted Difficulty (Easy / Medium / Hard)

Difficulty Score, a continuous value indicating relative complexity

The score is useful for understanding fine-grained difficulty differences.

Notes

Difficulty prediction is based only on text, so results are probabilistic.

Problem difficulty is subjective and may vary across platforms.

The application is designed to demonstrate a complete ML pipeline, from preprocessing to deployment.



AutoJudge Project  --  Rathod. Anshul ram