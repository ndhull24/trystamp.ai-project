import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import warnings
warnings.filterwarnings('ignore')

#NLTK stopwords 
nltk.download('stopwords')

# Loading the data
data = pd.read_excel(r"C:\Users\Navdeep\Desktop\VS CODE project\try-stamp\sample data.xlsx") #(I have shared the data file in the email, just change the path accordingly)

# Checking the class distribution of each header
print("Class distribution:")
print(data['label'].value_counts())
print("\n" + "="*50 + "\n")

# Combining the subject and body for classification input
data['text'] = data['subject'].fillna('') + ' ' + data['body'].fillna('')

# Removing the stopwords
stop = set(stopwords.words('english'))
data['text'] = data['text'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in stop]))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    data['text'], data['label'], test_size=0.2, stratify=data['label'], random_state=42
)

# Defining multiple models and hyperparameters to test and choose the best one
models_and_params = {
    'MultinomialNB': {
        'model': MultinomialNB(),
        'params': {
            'vect__max_features': [1000, 2000, None],
            'vect__ngram_range': [(1, 1), (1, 2)],
            'nb__alpha': [0.1, 1.0, 10.0]
        }
    },
    'LogisticRegression': {
        'model': LogisticRegression(max_iter=1000),
        'params': {
            'vect__max_features': [1000, 2000, None],
            'vect__ngram_range': [(1, 1), (1, 2)],
            'lr__C': [0.1, 1.0, 10.0],
            'lr__solver': ['liblinear', 'lbfgs']
        }
    },
    'RandomForest': {
        'model': RandomForestClassifier(n_estimators=100, random_state=42),
        'params': {
            'vect__max_features': [1000, 2000],
            'vect__ngram_range': [(1, 1), (1, 2)],
            'rf__n_estimators': [50, 100],
            'rf__max_depth': [10, None]
        }
    }
}

# Storing results for comparison
results = []

print("Testing different models and hyperparameters...\n")

# Testing each model
for model_name, config in models_and_params.items():
    print(f"Testing {model_name}...")
    
    # Created the pipeline for each model
    if model_name == 'MultinomialNB':
        pipeline = Pipeline([
            ('vect', TfidfVectorizer()),
            ('nb', config['model'])
        ])
    elif model_name == 'LogisticRegression':
        pipeline = Pipeline([
            ('vect', TfidfVectorizer()),
            ('lr', config['model'])
        ])
    else:  # RandomForest
        pipeline = Pipeline([
            ('vect', TfidfVectorizer()),
            ('rf', config['model'])
        ])
    
    # Grid search with cross-validation
    grid_search = GridSearchCV(
        pipeline, 
        config['params'], 
        cv=3, 
        scoring='accuracy',
        n_jobs=-1,
        verbose=0
    )
    
    # Fitting the grid search
    grid_search.fit(X_train, y_train)
    
    # Getting best model predictions
    best_model = grid_search.best_estimator_
    preds = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)

    # Storing results
    results.append({
        'model': model_name,
        'best_params': grid_search.best_params_,
        'accuracy': accuracy,
        'best_estimator': best_model
    })
    
    print(f"  Best accuracy: {accuracy:.4f}")
    print(f"  Best params: {grid_search.best_params_}")
    print()

# Finding the best model for the classification
best_result = max(results, key=lambda x: x['accuracy'])
best_model = best_result['best_estimator']

print("="*60)
print("FINAL RESULTS - MODEL COMPARISON")
print("="*60)

for result in results:
    print(f"{result['model']:<20}: {result['accuracy']:.4f}")

print(f"\nBEST MODEL: {best_result['model']} with accuracy: {best_result['accuracy']:.4f}")
print(f"BEST PARAMETERS: {best_result['best_params']}")
print("\n" + "="*60 + "\n")

# Evaluation of the best model
print("DETAILED EVALUATION OF BEST MODEL:")
print("-" * 40)
best_preds = best_model.predict(X_test)
print(classification_report(y_test, best_preds))

# Better classification function using the best model
def classify_email_advanced(subject, body):
    """
    Classify email using the automatically selected best model
    """
    email_text = (subject or '') + ' ' + (body or '')
    # Removing stopwords
    stop = set(stopwords.words('english'))
    email_text = ' '.join([word for word in email_text.split() if word.lower() not in stop])
    
    # Prediction and confidence
    pred = best_model.predict([email_text])[0]
    pred_proba = best_model.predict_proba([email_text])[0]
    confidence = max(pred_proba)
    
    return {
        'category': pred,
        'confidence': confidence,
        'model_used': best_result['model']
    }

# Testing again with the better function
print("\nTESTING ENHANCED CLASSIFICATION FUNCTION:")
print("-" * 45)
test_subject = "Invoice #456789"
test_body = "Attached is the invoice for services rendered in August."
result = classify_email_advanced(test_subject, test_body)

print(f"Email: '{test_subject}'")
print(f"Predicted category: {result['category']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Model used: {result['model_used']}")

# Final model results saved to CSV
results_df = pd.DataFrame({
    'text': X_test,
    'true_label': y_test,
    'predicted_label': best_preds
})
results_df['correct'] = results_df['true_label'] == results_df['predicted_label']
results_df.to_csv(r"C:\Users\Navdeep\Desktop\VS CODE project\try-stamp\best_model_results.csv", index=False)
print(f"\nDetailed results saved to: {r'C:\Users\Navdeep\Desktop\VS CODE project\try-stamp\best_model_results.csv'}")
