import json
import numpy as np
import time
from collections import Counter
from openai import OpenAI

OPENAI_MODEL_1 = "gpt-3.5-turbo"
OPENAI_MODEL_2 = "gpt-4o"

filename_base = str(int(time.time() * 1000))


def print_out(text):
    with open(filename_base + ".log", 'a') as log:
        log.write(text + "\n")
    print(text)


def get_gpt_response(prompt, api_key, model):
    client = OpenAI(
        api_key=api_key
    )

    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model=model
    )

    # Extract and return the response text
    return response.choices[0].message.content.strip()


ATTR_KEY_NUMBER = "number"
ATTR_KEY_LIST = "list"

prompt = f"""
What are the postulates of quantum mechanics?

Return answer in json object with following attributes:

{ATTR_KEY_NUMBER}: The numeric number of postulates you know about
{ATTR_KEY_LIST}: A list of strings where each item is one of the postulates

Just return the raw json and add no mark-up. Remove special characters text to ensure valid JSON.
"""

'''
E.g.

{
    "number_of_postulates": 5, 
    "postulates_list": [
        "The state of a quantum system can be fully described by a wave function", 
        "The evolution of a quantum system is governed by the Schr\u00f6dinger equation", 
        "Quantum measurements yield discrete and probabilistic outcomes", 
        "Quantum systems can exist in superposition of states", 
        "Entangled quantum systems can exhibit non-local correlations"
    ]
}
'''

print_out("Question:")
print_out("-----")

print_out(prompt)

print_out("-----")

api_key = "<API KEY>"  # Replace with your actual OpenAI API key

sampled_postulates_number = []
sampled_postulates_names = []
sampled_full_responses = []

NUMBER_OF_SAMPLES = 1000

# main

for model in [OPENAI_MODEL_1, OPENAI_MODEL_2]:

    print_out(f"Model: {model}")

    for i in range(0, NUMBER_OF_SAMPLES):

        response = get_gpt_response(prompt, api_key, model)

        try:
            response_json = json.loads(response)
        except:
            print_out("Error: Unable to parse and response. Skip.")
            continue

        sampled_full_responses.append(response_json)

        sampled_postulates_names.extend(response_json[ATTR_KEY_LIST])
        number_of_postulates = response_json[ATTR_KEY_NUMBER]

        sampled_postulates_number.append(number_of_postulates)

        print_out(f"Sample number {i}: {number_of_postulates}")

print_out("------ Done! -----")
average = np.mean(sampled_postulates_number)
print_out(f"Average: {average}")

print_out("")
counter = Counter(sampled_postulates_number)
count_dict = dict(counter)
print_out("Distribution of counts:")
print_out(str(json.dumps(count_dict, indent=4)))

print_out("")
counter = Counter(sampled_postulates_names)
count_dict = dict(counter)
print_out("Distribution of names:")
print_out(str(json.dumps(count_dict, indent=4)))

print_out("")
filename_json = filename_base + ".json"
print_out(f"Write all sampled responses to file: {filename_json}")
with open(filename_json, 'w') as json_file:
    json.dump(sampled_full_responses, json_file)
