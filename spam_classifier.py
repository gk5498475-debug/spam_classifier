"""
Spam Email Classifier
======================
Classifies emails/messages as "spam" or "ham" (not spam) using
TF-IDF text vectorization + Naive Bayes classification.

This script:
1. Uses a hand-crafted labeled dataset of spam/ham messages (swap in
   a real CSV with 'text' and 'label' columns if you have one, e.g.
   the classic SMS Spam Collection dataset)
2. Cleans and vectorizes text with TF-IDF
3. Trains a Multinomial Naive Bayes classifier (also compares Logistic Regression)
4. Evaluates with accuracy, precision, recall, F1, and a confusion matrix
5. Saves the trained model + vectorizer for reuse
6. Provides a predict_message() function for real-time predictions
"""

import re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import joblib

RANDOM_STATE = 42


def build_sample_dataset():
    """
    Hand-crafted labeled dataset of spam and ham messages.
    Replace this with: pd.read_csv("your_data.csv") -> columns ['text','label']
    for a real-world dataset.
    """
    spam_messages = [
        "Congratulations! You've won a $1000 Walmart gift card. Click here to claim now!",
        "URGENT: Your account has been suspended. Verify your details immediately.",
        "You have been selected for a FREE iPhone 15! Claim your prize now.",
        "Get rich quick! Work from home and earn $5000 a week, no experience needed.",
        "WINNER!! As a valued customer you have been selected to receive a cash prize.",
        "Limited time offer! Buy one get one free on all products. Click now!",
        "Your loan of $50000 has been approved. Reply now to receive funds instantly.",
        "Hot singles in your area are waiting to chat with you tonight!",
        "Lowest prices on Viagra and other meds, no prescription required, order now.",
        "Free entry into our weekly lottery draw, text WIN to 80085 to enter now.",
        "Congratulations, you've been chosen for a free cruise vacation! Call now.",
        "Act now! Your credit card has been charged $499, click to dispute immediately.",
        "Make money fast with this one simple trick banks don't want you to know!",
        "You have 1 new voicemail regarding your car's extended warranty, call now.",
        "Claim your free trial of our weight loss pills, results guaranteed in 7 days.",
        "Dear winner, you have won 2,000,000 in the international lottery, contact agent now.",
        "Increase your website traffic instantly! Buy followers and likes cheap today.",
        "Your PayPal account has unusual activity, verify your identity now to avoid suspension.",
        "Exclusive deal just for you: 90% off designer watches, limited stock, buy now!",
        "Congratulations! You qualify for a government grant of $10000, apply today.",
        "Nigerian prince needs your help transferring $10 million, you'll get a share.",
        "Final notice: your car warranty is about to expire, press 1 to renew now.",
        "Free gift card waiting for you, just complete this short survey to claim it.",
        "Your computer has a virus! Download our antivirus now to fix it immediately.",
        "Earn passive income trading crypto with our proven bot, sign up free today.",
    ]

    ham_messages = [
        "Hey, are we still meeting for lunch tomorrow at noon?",
        "Can you send me the report before end of day please?",
        "Happy birthday! Hope you have a wonderful day with family.",
        "The meeting has been rescheduled to 3pm on Thursday.",
        "Thanks for your help with the project, I really appreciate it.",
        "Don't forget to pick up milk and eggs on your way home.",
        "I attached the presentation slides for tomorrow's review.",
        "Let's catch up over coffee this weekend, I miss talking to you.",
        "The flight got delayed by two hours, I'll update you when I land.",
        "Great job on the presentation today, the client loved it.",
        "Can we reschedule our call to next week? Something came up.",
        "Here are the notes from today's team standup meeting.",
        "I'll be working from home tomorrow, let me know if you need anything.",
        "The kids' school play is on Friday evening, hope you can make it.",
        "Please review the attached invoice and confirm the amount is correct.",
        "Just checking in to see how you're doing after the surgery.",
        "Reminder: your dentist appointment is scheduled for 10am tomorrow.",
        "I found a great recipe for dinner tonight, want to try it together?",
        "The package was delivered this morning, thanks for ordering it.",
        "Looking forward to seeing you at the reunion next month.",
        "Can you review my code before I push it to the main branch?",
        "Let's plan the road trip itinerary this weekend, I have some ideas.",
        "Your subscription renewal was processed successfully, thank you.",
        "I left the documents on your desk, let me know if you need copies.",
        "The weather looks great this weekend, want to go hiking?",
    ]

    texts = spam_messages + ham_messages
    labels = ["spam"] * len(spam_messages) + ["ham"] * len(ham_messages)
    df = pd.DataFrame({"text": texts, "label": labels})
    return df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


def clean_text(text):
    """Basic text cleaning: lowercase, remove punctuation/numbers/extra spaces."""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def train_and_evaluate():
    df = build_sample_dataset()
    df["clean_text"] = df["text"].apply(clean_text)

    print(f"Dataset size: {len(df)} messages "
          f"({(df['label'] == 'spam').sum()} spam, {(df['label'] == 'ham').sum()} ham)\n")

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["label"], test_size=0.25,
        random_state=RANDOM_STATE, stratify=df["label"]
    )

    vectorizer = TfidfVectorizer(stop_words="english", max_features=3000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    models = {
        "Multinomial Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    }

    best_model, best_name, best_f1 = None, None, -1

    for name, clf in models.items():
        clf.fit(X_train_vec, y_train)
        preds = clf.predict(X_test_vec)

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, pos_label="spam")
        rec = recall_score(y_test, preds, pos_label="spam")
        f1 = f1_score(y_test, preds, pos_label="spam")

        print(f"{name}")
        print("-" * 40)
        print(f"Accuracy : {acc:.3f}")
        print(f"Precision: {prec:.3f}")
        print(f"Recall   : {rec:.3f}")
        print(f"F1 Score : {f1:.3f}")
        print(classification_report(y_test, preds))
        print()

        if f1 > best_f1:
            best_model, best_name, best_f1 = clf, name, f1

    print(f"Best model: {best_name} (F1 = {best_f1:.3f})")

    # Confusion matrix plot for best model
    preds_best = best_model.predict(X_test_vec)
    cm = confusion_matrix(y_test, preds_best, labels=["ham", "spam"])
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title(f"Confusion Matrix - {best_name}")
    plt.colorbar()
    plt.xticks([0, 1], ["ham", "spam"])
    plt.yticks([0, 1], ["ham", "spam"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
    plt.tight_layout()
    plt.savefig("spam_confusion_matrix.png", dpi=150)
    print("Saved: spam_confusion_matrix.png")

    joblib.dump(best_model, "spam_classifier_model.joblib")
    joblib.dump(vectorizer, "spam_vectorizer.joblib")
    print("Saved: spam_classifier_model.joblib, spam_vectorizer.joblib")

    return best_model, vectorizer


def predict_message(model, vectorizer, message):
    """Predict whether a single message is spam or ham."""
    cleaned = clean_text(message)
    vec = vectorizer.transform([cleaned])
    prediction = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]
    classes = model.classes_
    prob_dict = dict(zip(classes, proba))
    return prediction, prob_dict


if __name__ == "__main__":
    model, vectorizer = train_and_evaluate()

    print("\n--- Example Predictions ---")
    test_messages = [
        "Congratulations, you have won a free vacation, click to claim now!",
        "Hey, are you free for dinner tonight?",
        "URGENT: verify your bank account now or it will be suspended",
        "Can you send me the meeting notes from yesterday?",
    ]
    for msg in test_messages:
        pred, proba = predict_message(model, vectorizer, msg)
        print(f"'{msg}'\n  -> Prediction: {pred.upper()} | Confidence: {proba[pred]:.2%}\n")
