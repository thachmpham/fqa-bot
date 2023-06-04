from query import *

import openai

# Set up your OpenAI API credentials
openai.api_key = 'YOUR_API_KEY'

# Define the function to answer questions
def answer_question(question, knowledge_base):
    # Prepare the input by concatenating the question and knowledge base documents
    input_text = f"Question: {question}\nContext: {knowledge_base}\n"

    # Generate the answer using OpenAI's completions API
    response = openai.Completion.create(
        engine='text-davinci-003',
        prompt=input_text,
        max_tokens=100,  # Adjust the value based on your answer length requirements
        n=1,             # Set the number of completions to generate
        stop=None,       # You can provide a stopping criterion if needed
        temperature=0.7, # Adjust the temperature for response randomness
        top_p=1.0        # Adjust the top_p value for diversity in responses
    )

    # Extract and return the generated answer from the model response
    answer = response.choices[0].text.strip()
    return answer

# Example usage
knowledge_base = '''
The knowledge base contains relevant information in a structured format.
You can format the information as separate paragraphs, bullet points, or any other suitable format.
Make sure to provide enough context for the model to understand the knowledge base.

Here's an example of a knowledge base entry:
- Title: OpenAI
  Description: OpenAI is an artificial intelligence research laboratory.
'''

question = "What is OpenAI?"
answer = answer_question(question, knowledge_base)
print("Question:", question)
print("Answer:", answer)