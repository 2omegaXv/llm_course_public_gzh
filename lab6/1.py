from dotenv import load_dotenv
import threading
import evaluate
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random as rd
from tqdm import tqdm
import os
import time
import random
from openai import RateLimitError
from openai import OpenAI
from datasets import load_dataset
load_dotenv()
openai_api_key = os.environ.get("INFINI_API_KEY")
openai_base_url = os.environ.get("INFINI_BASE_URL")

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_base_url,
)
# a single thread version of get_llm_output
def get_llm_output(model_name, question_content, question_id, output, semaphore=None):
    # the last semaphore is used for parallel execution only
    try:
        retries = 5
        for attempt in range(retries):
            try:
                if isinstance(question_content, str):
                    messages = [{"role": "user", "content": question_content}]
                else:
                    messages = [{"role": "user", "content": q} for q in question_content]

                chat_response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    max_tokens=2048,
                    temperature=0,
                    seed=42
                )

                llm_answer = chat_response.choices[0].message.content.strip()
                output[question_id] = llm_answer
                break  # 成功就退出 retry 循环

            except RateLimitError as e:
                wait_time = random.uniform(1, 3) * (2 ** attempt)
                print(f"RateLimitError on Q{question_id}, retry {attempt+1}/{retries}, wait {wait_time:.1f}s")
                time.sleep(wait_time)

            except Exception as e:
                print(f"Error on Q{question_id}: {e}")
                break  # 其他异常不重试

        else:
            # 所有尝试失败，填空防止后续崩
            output[question_id] = ""

    finally:
        if semaphore:
            semaphore.release() # release the semaphore

# extending the single thread version to parallel execution
def get_llm_output_parallel(model_name, question_contents, max_threads=5):
    # Create threads for each question
    output = {}
    threads = []
    semaphore = threading.Semaphore(max_threads)
    for question_id, question_content in tqdm(enumerate(question_contents)):
        semaphore.acquire() 
        thread = threading.Thread(target=get_llm_output, args=(model_name, question_content, question_id, output, semaphore))
        threads.append(thread)
        thread.start()
        # semaphore is released when the thread ends, in the single thread version

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    sorted_keys = sorted(output.keys())
    sorted_outputs = [output[key] for key in sorted_keys]        
    return sorted_outputs

def print_llm_outputs(model_name, question_contents, llm_answers, references):
    for i, (question, answer, reference) in enumerate(zip(question_contents, llm_answers, references)):
        print('Question %d: %s'%(i, question))
        print('Answer from Model %s: %s'%(model_name, answer))
        print('Reference Answer: %s\n'%(reference))

def get_options(llm_answers):
    # Select the option that occurs most times in the model output as the final answer.
    options = []
    for llm_answer in llm_answers:
        option_frequencies = [llm_answer.count(option) for option in 'ABCD']
        most_frequent = np.argmax(option_frequencies)
        most_frequent_option = 'ABCD'[most_frequent]
        options.append(most_frequent_option)
    return options

def option2num(options):
    # Transform the ABCD options to numbers for accuracy evaluation.
    option2num_dict = {'A':0 ,'B':1, 'C':2, 'D':3}
    nums = list(map(lambda x:option2num_dict[x], options))
    return nums
accuracy = evaluate.load("accuracy")

# Load GPQA dataset
gpqa_dataset = load_dataset("csv", data_files={"gpqa": "/ssdshare/share/data/gpqa/gpqa_main.csv"})["gpqa"]

# Prepare question contents and answers
question_contents = gpqa_dataset['question']
answers = gpqa_dataset['answer']

overall_scores = {}
for model_name in ['qwen2.5-72b-instruct', 'llama-3.3-70b-instruct']:
    print(f'============== {model_name}  ==============')
    print(f'GPQA has {len(question_contents)} questions')
    llm_answers = get_llm_output_parallel(model_name, question_contents, max_threads=5)
    print_llm_outputs(model_name, question_contents, llm_answers, answers)
    
    # Assuming answers in GPQA are already in 'ABCD' format or similar
    llm_answers = get_options(llm_answers)
    
    acc = accuracy.compute(references=option2num(answers), predictions=option2num(llm_answers))
    overall_scores[model_name] = [acc['accuracy']]  # Wrap in a list to match the DataFrame structure

accuracy_df = pd.DataFrame(overall_scores)
accuracy_df.index = ['GPQA']  # Set index to dataset name
print(accuracy_df)

accuracy_df.plot.barh()
plt.xlabel('Accuracy')
plt.title('LLM performance on GPQA')
plt.show()
