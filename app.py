import streamlit as st
import tensorflow as tf
import pickle

#changes
def preprocess_text(text):

    #text = text.str.lower()  # converting to lowercase
    #text = text.str.replace('[^\w\s]', '')  # removing punctuation

    with open('tokenizer.pickle', "rb") as f:
        tokenizer=pickle.load(f)

    text=[text]
    sequences = tokenizer.texts_to_sequences(text)
    padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=267)

    return padded_sequence


def predict_sentiment(text, model):
    # Preprocess the tweet
    preprocessed_text = preprocess_text(text)

    # Classify the tweet using the classifier
    sentiment = model.predict(preprocessed_text)

    return sentiment
def main():
    st.title("Semi Supervised sequence learning")
    text=st.text_input("Enter text to generate sentiment")
    if st.button("Generate"):
        if text:
            model = tf.keras.models.load_model('lstm_model.h5')
            sentiment=predict_sentiment(text, model)
            if sentiment>0.5:
                st.write("Predicted Sentiment: Positive")
            else:
                st.write("Predicted Sentiment: Negative")
        else:
            st.warning("Please enter text to generate sentiment")


if __name__=="__main__":
    main()