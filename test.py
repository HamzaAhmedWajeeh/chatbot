import openai
from langdetect import detect

llm_model = "gpt-3.5-turbo"

def is_english(text):
    try:
        return detect(text) == "en"
    except:
        return False

def get_completion(prompt, model=llm_model):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that acts as a medical chatbot. Users will enter symptoms they are facing in natural language. You should reply with possible diseases, suggest precautions, and recommend seeing a doctor if necessary. Never say that YOU ARE NOT A DOCTOR. Only respond to queries related to medical and health, else just say that I am a medical chat bot trained only on medical diseases data and I can't help with anyother query. Respond in max 3 lines. BRIEFLY. If the request is not in English language, or in another language but English alphabets are used, such as Roman Urdu, respond with 'Only English language is supported.' Only respond to prompt which words are present in English Dictionary. Only respond in English Language.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=150,
    )
    # if not is_english(prompt):
    #     return "Only English language is supported"
    return response.choices[0].message["content"]
