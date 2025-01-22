import streamlit as st
import pickle

model = pickle.load(open('spam.pkl','rb'))
cv = pickle.load(open('vectorizer.pkl','rb'))

st.title("SMS Spam Classification Application")
st.write("Introducing a powerful machine learning application specifically designed to effectively classify SMS as spam or non-spam.")

user_input= st.text_area("Enter SMS to classify",height=150)

if st.button("Classify") :
    if user_input:
        data = [user_input]
        vectorized_data = cv.transform(data).toarray()
        result = model.predict(vectorized_data)
        if result[0]==0:
            st.write("The SMS is not spam")
        else:
            word_count = len(user_input.split())
            st.write("The SMS is spam")
            st.write(f"The SMS contains {word_count} words.")
            
    else:
        st.write("Please type SMS to classify")