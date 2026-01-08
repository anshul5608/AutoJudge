**AutoJudge**
AutoJudge is a machine learning–based application that predicts the difficulty level of programming problems using only their textual descriptions.
It classifies problems into Easy, Medium, or Hard and also provides a numerical difficulty score for finer difficulty estimation.


**Project Overview**
Competitive programming platforms label problems by difficulty, but these labels are often subjective and vary across platforms.
AutoJudge attempts to automate difficulty prediction by learning patterns from problem statements using machine learning.
The project demonstrates a complete end-to-end ML pipeline, including preprocessing, feature extraction, model training, evaluation, and deployment through a web interface.


**Dataset Used**
The dataset used for this project is stored in:
data/problems_data.jsonl

Each dataset entry contains:
  -Problem title
  -Problem description
  -Input and output specifications
  -Difficulty class (Easy / Medium / Hard)
  -Numerical difficulty score
The dataset includes problems from competitive programming platforms and reflects real-world subjectivity in difficulty labeling.


**Approach and Models Used**
*Preprocessing*
  -Text cleaning and normalization
  -Merging description, input format, and output format into a single text

*Feature Extraction*
 -TF-IDF (Term Frequency–Inverse Document Frequency) for text representation
 -Simple numeric features (text length, word count) for difficulty score prediction

*Models*
  -Classifier: Linear Support Vector Classifier (LinearSVC)
    Used to predict difficulty category (Easy / Medium / Hard)
  -Regressor: Random Forest Regressor
    Used to predict a continuous difficulty score


**Evaluation Metrics**
*Classification* 
  -Accuracy evaluated on a held-out test set
  -Random baseline accuracy for three classes is 33%
  -Achieved accuracy is significantly above random baseline

*Regression*
  -MAE (Mean Absolute Error)
  -RMSE (Root Mean Squared Error)
These metrics measure how close the predicted difficulty score is to the actual score.


**Steps to Run the Project Locally**
  1. Install Dependencies
    pip install -r requirements.txt

  2. Generate Features
    python src/features.py

  3. Train Models
    python src/train_classifier.py
    python src/train_regressor.py

  4. Run the Web Application
    python web/app.py

  Open your browser and visit:
    http://127.0.0.1:5000


**Web Interface Explanation**
The web interface allows users to:
Paste a programming problem statement
Get a predicted difficulty class (Easy / Medium / Hard)
View a numerical difficulty score
The interface also handles low-confidence predictions and short inputs to improve usability and reliability.


**Demo Video**
  https://drive.google.com/drive/folders/1SRkyDWGsUirLwVx-um2PtKrVte9u33Xc?usp=sharing

**Details**
Name: Rathod Anshul Ram
Institute: Indian Institute of Technology Roorkee
Branch: Computer Science and Engineering - 2nd Year
En.No : 24114074
