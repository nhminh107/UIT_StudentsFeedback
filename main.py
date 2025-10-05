from datasets import load_dataset
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from underthesea import word_tokenize
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

def tokenize_vietnamese_text(text):
    """Tokenize Vietnamese text"""
    if pd.isna(text):
        return ""
    return word_tokenize(str(text), format="text")


# Tải bộ dữ liệu
dataset = load_dataset("uitnlp/vietnamese_students_feedback")
df_train = dataset['train'].to_pandas()
df_train['sentence'] = df_train['sentence'].apply(tokenize_vietnamese_text)
df_validation = dataset['validation'].to_pandas()
df_validation['sentence'] = df_validation['sentence'].apply(tokenize_vietnamese_text)
df_test = dataset['test'].to_pandas()
df_test['sentence'] = df_test['sentence'].apply(tokenize_vietnamese_text)

X_train = df_train.drop(['sentiment'], axis=1)
X_validation = df_validation.drop(['sentiment'], axis=1)
X_test = df_test.drop(['sentiment'], axis=1)
Y_train = df_train['sentiment']
Y_validation = df_validation['sentiment']
Y_test = df_test['sentiment']

# Pipeline cho text
sentence_pipeline = Pipeline(steps=[
    ('TFIDF', TfidfVectorizer(
        token_pattern=r'(?u)\b\w+\b',
        smooth_idf=True,
        max_features=5000
    ))
])

# Pipeline cho categorical features
numeric_prep = Pipeline(steps=[
    ('enc', OneHotEncoder(handle_unknown='ignore', sparse_output=True))
])

print("Topic unique values:", X_train['topic'].unique())

# ColumnTransformer
ct = ColumnTransformer([
    ("sentencePre", sentence_pipeline, "sentence"),
    ("numPre", numeric_prep, ["topic"])
], sparse_threshold=1.0)

# GridSearch parameters
parameters = {
    'penalty': [ 'l2'],
    'C': [0.1, 1.0, 10.0, 100.0],
}

print("Bắt đầu vector hóa dữ liệu...")
X_train_vector = ct.fit_transform(X_train)
print(f"Vector hóa hoàn tất. Kích thước X_train: {X_train_vector.shape}")

print("\nBắt đầu tinh chỉnh tham số...")
grid_search = GridSearchCV(
    LogisticRegression(random_state=42, max_iter=1000),
    parameters,
    cv=2,
    scoring='f1_macro',
    n_jobs=-1,
    verbose=2
)
grid_search.fit(X_train_vector, Y_train)
print("Tinh chỉnh hoàn tất.")

# In kết quả tốt nhất
print("\n=================== KẾT QUẢ TỐT NHẤT ===================")
print(f"Tham số tốt nhất: {grid_search.best_params_}")
print(f"F1 Score tốt nhất (trên CV): {grid_search.best_score_:.4f}")

# Lấy best model
best_model = grid_search.best_estimator_
print(f"\nMô hình tốt nhất: {best_model}")

# Đánh giá trên validation set
print("\n=================== ĐÁNH GIÁ TRÊN VALIDATION SET ===================")
X_validation_vector = ct.transform(X_validation)
y_pred_val = best_model.predict(X_validation_vector)
print(f"Accuracy: {accuracy_score(Y_validation, y_pred_val):.4f}")
print("\nClassification Report:")
print(classification_report(Y_validation, y_pred_val))

# Đánh giá trên test set
print("\n=================== ĐÁNH GIÁ TRÊN TEST SET ===================")
X_test_vector = ct.transform(X_test)
y_pred_test = best_model.predict(X_test_vector)
print(f"Accuracy: {accuracy_score(Y_test, y_pred_test):.4f}")
print("\nClassification Report:")
print(classification_report(Y_test, y_pred_test))