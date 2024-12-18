import streamlit as st
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

def load_model(task):
    if task == "Text Summarization":
        return pipeline("summarization")
    elif task == "Next Word Prediction":
        model_name = "gpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        return tokenizer, model
    elif task == "Story Prediction":
        return pipeline("text-generation", model="gpt2")
    elif task == "Chatbot":
        return pipeline("conversational", model="microsoft/DialoGPT-medium")
    elif task == "Sentiment Analysis":
        return pipeline("sentiment-analysis")
    elif task == "Question Answering":
        return pipeline("question-answering")
    elif task == "Image Generation":
        return pipeline("stable-diffusion")  # Ensure this model is available.
    else:
        return None

def main():
    st.title("AI Multi-Task Application")
    task = st.sidebar.selectbox("Choose a task", ["Text Summarization", "Next Word Prediction", "Story Prediction",
                                                  "Chatbot", "Sentiment Analysis", "Question Answering", "Image Generation"])
    user_input = st.text_area("Enter your input:")

    if st.button("Generate Output"):
        if task == "Text Summarization" and user_input:
            summarizer = load_model(task)
            summary = summarizer(user_input, max_length=100, min_length=30, do_sample=False)
            st.write("**Summary:**", summary[0]['summary_text'])
        elif task == "Next Word Prediction" and user_input:
            tokenizer, model = load_model(task)
            input_ids = tokenizer.encode(user_input, return_tensors='pt')
            outputs = model.generate(input_ids, max_length=len(input_ids[0]) + 3, num_return_sequences=1)
            prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
            st.write("**Next Word Prediction:**", prediction)
        elif task == "Story Prediction" and user_input:
            story_gen = load_model(task)
            story = story_gen(user_input, max_length=150, num_return_sequences=1)
            st.write("**Generated Story:**", story[0]['generated_text'])
        elif task == "Chatbot" and user_input:
            chatbot = load_model(task)
            response = chatbot(user_input)
            st.write("**Chatbot Response:**", response.generated_responses[-1])
        elif task == "Sentiment Analysis" and user_input:
            sentiment = load_model(task)
            result = sentiment(user_input)
            st.write("**Sentiment Analysis Result:**", result[0])
        elif task == "Question Answering" and user_input:
            qa = load_model(task)
            context = st.text_area("Enter the context for the question:")
            if context:
                answer = qa(question=user_input, context=context)
                st.write("**Answer:**", answer['answer'])
        elif task == "Image Generation" and user_input:
            image_gen = load_model(task)
            image = image_gen(user_input)
            st.image(image, caption="Generated Image", use_column_width=True)
        else:
            st.warning("Please enter valid input for the selected task.")

if __name__ == "__main__":
    main()
