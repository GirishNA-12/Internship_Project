from transformers import QuestionAnsweringPipeline, pipeline

def question_answer(context: str, question: str):
    qa_pipeline = pipeline("question-answering")
    result = qa_pipeline(question=question, context=context)
    return result['answer']
