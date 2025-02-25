from model.model import question_answer
import streamlit as st
import json

# Title
st.title("Contextual Language Understanding with Transformer Models: Elevating NLP Capabilities")

# Input for context

context = st.text_area("## Enter the context here:",
                       placeholder="Type the context, description, or background...")

# Input for question
question = st.text_input("Enter your question:",
                         placeholder="What would you like to ask based on the context provided?")

# Button to submit
if st.button("Submit"):
    # Display the inputs
    if context and question:

        result = question_answer(context, question);

        prompt = {"context": context, "question": question}
        json_string = json.dumps(prompt)

        with open("prompt.json", "w") as f: 
            f.write(json_string)

        st.subheader("Result:")
        st.info(result)
    else:
        st.warning("Please fill in both the context and the question!")
