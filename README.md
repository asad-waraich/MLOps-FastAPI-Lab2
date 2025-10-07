# MLOPS Course: Lab 2 - SVM Model with FastAPI

This project is an independent lab in the MLOPS course, demonstrating how to serve a machine learning model as a REST API using Python, FastAPI, and Scikit-Learn. This lab was self-initiated to gain hands-on experience with FastAPI and model deployment.

The API is built to predict the class of a given sample based on four input features. The model is a **Support Vector Machine (SVM) Classifier** trained on a synthetically generated dataset to simulate a real-world machine learning application.

---

## 🚀 Project Workflow

1. **Data Generation**: A synthetic dataset with 200 samples, 4 features, and 3 distinct classes is generated using `sklearn.datasets.make_classification`.
2. **Data Visualization**: The script creates and saves a pair plot of the features to help understand their distributions and relationships.
3. **Model Training**: An `SVM Classifier` with RBF kernel is trained on the generated data and saved to a file (`synthetic_model.pkl`).
4. **API Serving**: A FastAPI server loads the trained model and exposes a `/predict` endpoint to serve predictions over HTTP.


---

## 📁 Project Structure

```
.
├── model/
│   └── synthetic_model.pkl
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── main.py
│   ├── predict.py
│   ├── train.py
│   └── visualize.py
├── .gitignore
├── README.md
└── requirements.txt
```

---

## ⚙️ How to Run This Project

### 1. Setup the Environment

First, create and activate a Python virtual environment.

```bash
# Create the environment
python3 -m venv .venv

# Activate the environment (macOS/Linux)
source .venv/bin/activate

# Or on Windows
# .\.venv\Scripts\activate
```

### 2. Install Dependencies

Install all the required packages from the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 3. Visualize the Data (Optional but Recommended)

Navigate to the `src` directory and run the visualization script. This will create the `assets` folder and save the `data_distribution.png` plot inside it.

```bash
cd src
python visualize.py
```

### 4. Train the Model

From the `src` directory, run the training script. This will create the `model` folder and save the `synthetic_model.pkl` file.

```bash
python train.py
```

### 5. Run the API Server

Finally, start the FastAPI server using Uvicorn.

```bash
uvicorn main:app --reload
```

The API will now be running on `http://127.0.0.1:8000`.

---

## 🧪 Testing the API

You can test the API using either the interactive documentation or a `curl` command.

### Using the Browser (Recommended)

1. Open your web browser and go to **http://127.0.0.1:8000/docs**
2. You'll see the Swagger UI with interactive API documentation
3. Click on `POST /predict` endpoint to expand it
4. Click "Try it out"
5. Fill in the four feature fields in the request body
6. Click "Execute"

### Using Curl

Open a new terminal and run the following command to test the endpoint:

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "feature1": 1.5,
  "feature2": -0.8,
  "feature3": 2.1,
  "feature4": 0.3
}'
```

You will receive a JSON response containing the model's prediction, such as:

```json
{
  "prediction": 1
}
```

---

## 📊 Data Visualization

The `visualize.py` script generates a pair plot to show the relationships between the features and how the different classes are distributed. This is a critical step for understanding the dataset before modeling. The output plot is saved in the `assets/` directory.

---

## 🤖 About the Model

This lab uses a **Support Vector Machine (SVM)** classifier with an RBF (Radial Basis Function) kernel. SVM is a powerful supervised learning algorithm that works well for classification tasks, especially with clear margins of separation between classes. The RBF kernel allows the SVM to handle non-linearly separable data by mapping it to a higher-dimensional space.

Key characteristics:
- **Algorithm**: Support Vector Machine (SVC)
- **Kernel**: RBF (Radial Basis Function)
- **Dataset**: 200 synthetic samples with 4 features
- **Classes**: 3 distinct classes (0, 1, 2)
- **Random State**: 42 (for reproducibility)

---

## 🎯 Learning Objectives

This self-directed lab was created to:
- Gain hands-on experience with **FastAPI** for model serving
- Understand the end-to-end ML pipeline from training to deployment
- Practice creating RESTful APIs for machine learning models
- Learn to use **Swagger UI** for API documentation and testing
- Explore data visualization techniques for understanding datasets
- Implement proper project structure for ML applications

---

## 📝 Requirements

See `requirements.txt` for the full list of dependencies, which includes:
- FastAPI
- Uvicorn
- Scikit-learn
- Pandas
- Matplotlib
- Seaborn
- NumPy
- Joblib