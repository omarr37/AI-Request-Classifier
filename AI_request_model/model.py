import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler
import joblib

from preprocess import preprocess_message


# تحميل البيانات
df = pd.read_csv("C:/AI_Projects/my_python/AI_request_model/dataset.csv")
df["category"] = df["category"].str.strip().replace(r"\s+", " ", regex=True)
df.drop_duplicates(inplace=True)
df.dropna(subset=["message", "category"], inplace=True)

# تطبيق التنظيف على كل الرسائل
df["message_clean"] = df["message"].apply(preprocess_message)

X = df["message_clean"]
y = df["category"].astype(str)

ros = RandomOverSampler(random_state=42)
X_res, y_res = ros.fit_resample(X.values.reshape(-1, 1), y)
X_res = X_res.ravel()

X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# بناء النموذج
model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,2))),
    ("clf", RandomForestClassifier(n_estimators=200, random_state=42))
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

joblib.dump(model, "ai_request_model.pkl")
print("✅ Model trained and saved as ai_request_model.pkl")

import joblib

def load_model():
    model = joblib.load("ai_request_model.pkl")  # Contains both vectorizer + model
    vectorizer = model.named_steps["tfidf"]
    return model, vectorizer

from sklearn.metrics import classification_report
import pandas as pd

from sklearn.metrics import classification_report
import pandas as pd

# تقييم النموذج
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()

# حفظ التقرير
report_df.to_csv("classification_report.csv", index=True)
print("📊 Classification report saved as classification_report.csv")

