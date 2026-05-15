---
title: AI News Classifier
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# AI News Classification Project

## 1. Project Description
This project is a machine learning-based web application that classifies news statements as REAL or FAKE using Natural Language Processing techniques.

The system takes user input (text news) via a REST API and predicts whether the news is real or fake using a trained machine learning model.

---

## 2. Technologies Used
- Python
- Flask
- Scikit-learn
- Joblib
- HTML/CSS (optional frontend)

---

## 3. Model Description
- TF-IDF Vectorizer
- Linear Support Vector Machine (SVM)
- Accuracy: ~72%

---

## 4. Input and Output Format

### Input:
User sends a JSON request containing a news statement.

Example Input:
{
  "news": "Government increases education budget"
}

### Output:
The system returns a JSON response:

{
  "prediction": "REAL"
}

or

{
  "prediction": "FAKE"
}

---

## 5. Features
- Real-time news classification API
- JSON-based REST API using Flask
- Machine learning prediction system
- Unit testing using unittest framework
- Threading implementation for request handling

---

## 6. How to Run Project

1. Install dependencies:
pip install -r requirements.txt

2. Train the model:
python train_model.py

3. Run Flask app:
python app.py

4. Open API locally:
http://127.0.0.1:7860

---

## 7. API Testing (Postman)

- Method: POST  
- URL: http://127.0.0.1:7860/predict  

### Body Type:
raw → JSON (application/json)

### Example Request:
{
  "news": "Aliens landed in Karachi yesterday according to secret sources"
}

### Example Response:
{
  "prediction": "FAKE"
}

---

## 8. Unit Testing

Run tests using:
python testing.py

This verifies that:
- Real news is classified correctly
- Fake news is classified correctly

---

## 9. Project Management (Trello)

A Trello board was used to manage the development process:
- Model training
- API development
- Testing
- Deployment tasks

---

## 10. Deployment

- GitHub Repository:
- Hugging Face Deployment: 

---

## 11. Credits

This project is inspired by open-source machine learning projects available on GitHub.  
The implementation has been extended with software engineering concepts such as:

- API development using Flask
- Unit testing
- Threading
- REST API integration
- Deployment preparation

---

## 12. Future Improvements
- Improve accuracy using deep learning models
- Use larger and balanced dataset
- Add authentication system
- Deploy on cloud platforms (AWS / Render)
