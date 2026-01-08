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
    

  3. Generate Features
     
    python src/features.py


  5. Train Models
     
    python src/train_classifier.py
    
    python src/train_regressor.py


  7. Run the Web Application
     
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




Here is the complete `README.md` content, professionally formatted with visual hierarchy to make your project stand out.

```markdown
# 🤖 AutoJudge: ML-Powered Problem Difficulty Predictor

**AutoJudge** is a machine learning–based application designed to predict the difficulty level of programming problems using only their textual descriptions. It bridges the gap between subjective platform labeling and objective complexity analysis.

---

## 🚀 Project Overview
Competitive programming platforms label problems by difficulty, but these labels are often subjective and vary across sites. **AutoJudge** automates this process by learning linguistic and structural patterns from problem statements.

This project demonstrates a complete **end-to-end ML pipeline**, including:
* *Preprocessing & Text Normalization*
* *Feature Extraction (TF-IDF & Numeric)*
* *Model Training & Evaluation*
* *Web Deployment via Flask*

---

## 📊 Dataset Specifications
The model is trained on a curated dataset located at `data/problems_data.jsonl`. Each entry includes:
* **Problem Metadata:** Title, Description, Input/Output specifications.
* **Target Labels:** Difficulty Class (*Easy, Medium, Hard*) and a numerical difficulty score for fine-grained estimation.

---

## 🧠 Approach and Models

### 1. Preprocessing & Feature Extraction
* **Text Cleaning:** Normalization of problem statements and merging fields (Description + Input + Output) into a unified corpus.
* **TF-IDF Vectorization:** Used to convert text into a high-dimensional representation of term importance.
* **Numeric Features:** Includes text length and word count to provide context for the regression model.

### 2. Machine Learning Models


[Image of machine learning workflow diagram]

* **Classification:** *Linear Support Vector Classifier (LinearSVC)*
    * Predicts the categorical label: **Easy, Medium, or Hard**.
* **Regression:** *Random Forest Regressor*
    * Predicts a **continuous numerical score** for precise difficulty mapping.

### 3. Evaluation Metrics
* **Classification:** Accuracy (Significantly outperforms the 33% random baseline).
* **Regression:** Evaluated using **MAE** (Mean Absolute Error) and **RMSE** (Root Mean Squared Error).

---

## 🛠️ Installation & Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt

```

### 2. Feature Engineering

```bash
python src/features.py

```

### 3. Model Training

```bash
python src/train_classifier.py
python src/train_regressor.py

```

### 4. Launch the Web Interface

```bash
python web/app.py

```

*Open your browser and visit: **http://127.0.0.1:5000***

---

## 🌐 Web Interface

The interactive UI allows users to paste any programming problem statement to receive:

1. **Predicted Difficulty Class** (*Easy / Medium / Hard*)
2. **Numerical Difficulty Score**
3. **Confidence Handling:** Built-in logic to handle short inputs or low-confidence text.

---

## 🎥 Project Demo

**[Click here to view the Demo Video](https://drive.google.com/drive/folders/1SRkyDWGsUirLwVx-um2PtKrVte9u33Xc?usp=sharing)**

---

## 👤 Developer Information

* **Name:** Rathod Anshul Ram
* **Institute:** Indian Institute of Technology Roorkee (IITR)
* **Branch:** Computer Science and Engineering — 2nd Year
* **Enrollment No:** 24114074

```

***

**Would you like me to add a "Future Scope" section or include a sample table for the evaluation metrics?**

```
