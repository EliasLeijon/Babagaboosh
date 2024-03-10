model_examples = ["google/flan-t5-large", "microsoft/phi-2", "google/flan-t5-xl", "NousResearch/Nous-Hermes-2-Mistral-7B-DPO"]
used_model = 3

MODEL_DOWNLOAD_DIR = model_examples[used_model]
MODEL_SAVE_DIR = "openSourceModels/"+model_examples[used_model].split("/")[-1]

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline, LlamaTokenizer, MixtralForCausalLM, AutoModelForCausalLM
from langchain.llms import HuggingFacePipeline
import torch
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from transformers import BitsAndBytesConfig
from colorama import Fore, Back, Style




class OpenSourceManager:
    text_gen_pipeline = None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Stores the entire conversation format is dict(role: string, content: string)
    chat_history = []
    

    def __init__(self):

        try: 
            # model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_SAVE_DIR)
            # tokenizer = AutoTokenizer.from_pretrained(MODEL_SAVE_DIR)

            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_SAVE_DIR, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_SAVE_DIR,
                torch_dtype=torch.float16,
                device_map="auto",
                load_in_8bit=False,
                load_in_4bit=True,
                use_flash_attention_2=False,
            )

        except:
            print("Could not find model, downloading and saving to disk")
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DOWNLOAD_DIR, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_DOWNLOAD_DIR,
                torch_dtype=torch.float16,
                device_map="auto",
                load_in_8bit=False,
                load_in_4bit=True,
                use_flash_attention_2=False,
            )
            self.tokenizer.save_pretrained(MODEL_SAVE_DIR)
            model.save_pretrained(MODEL_SAVE_DIR)

        pipe = pipeline(
            "text2text-generation",  # Specify the task as text-to-text generation
            model=model,              # Use the previously initialized model
            tokenizer=self.tokenizer,      # Use the previously initialized tokenizer
            max_length=8000,          # Set the maximum length for generated text to 512 tokens
            temperature=0.3,          # Set the temperature parameter for controlling randomness (0 means deterministic)
            top_p=0.95,               # Set the top_p parameter for controlling the nucleus sampling (higher values make output more focused)
            repetition_penalty=1.15,   # Set the repetition_penalty to control the likelihood of repeated words or phrases
            # device=self.device,
            do_sample=True,
            pad_token_id = self.tokenizer.pad_token_id
        )

        self.text_gen_pipeline = pipe # HuggingFacePipeline(pipeline=pipe)
 

    def _num_tokens_in_history(self):
        return len(self.tokenizer.apply_chat_template(self.chat_history, return_tensors='pt')[0])

    def chat(self, prompt):
        prompt = self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False)
        print(Fore.GREEN, self.text_gen_pipeline(
            prompt, do_sample=True, top_k=10
            )[0]["generated_text"][len(prompt):])
        print(Fore.WHITE)


    def chat_with_history(self, prompt):
        self.chat_history.append({"role": "user", "content": prompt})
        # print(Fore.BLUE+f"Current tokens in history {self._num_tokens_in_history()}")
        full_history_prompt = self.tokenizer.apply_chat_template(self.chat_history, tokenize=False)+"<|im_start|>assistant\n"
        # print(Fore.YELLOW+full_history_prompt)
        model_output =  self.text_gen_pipeline(full_history_prompt, return_text=False)[0]["generated_text"].split("assistant\n")[-1]
        self.chat_history.append({"role": "assistant", "content": model_output})
        print(Fore.GREEN, model_output)
        print(Fore.WHITE)


if __name__ == '__main__':
    opensource_manager = OpenSourceManager()

    # CHAT WITH HISTORY TEST
    # FIRST_SYSTEM_MESSAGE ={"role":'system', "content": '''
    # You are Pajama Sam, the lovable protagonist from the children's series Pajama Sam from Humongous Entertainment. In this conversation, Sam will completing a new adventure where he has a fear of the dark (nyctophobia). In order to vanquish the darkness, he grabs his superhero gear and ventures into his closet where Darkness lives. After losing his balance and falling into the land of darkness, his gear is taken away by a group of customs trees. Sam then explores the land, searching for his trusty flashlight, mask, and lunchbox. 
                            
    # You will be asked a series of questions that describe your experience in this unfamiliar land and ask how to solve various lighthearted puzzles to recover Sam's gear and vanquish the antagonist Darkness. 
                            
    # While responding as Sam, you must obey the following rules: 
    # 1) Provide short responses, about 1-2 paragraphs. 
    # 2) Always stay in character, no matter what. 
    # 3) Occasionally make loud noises consisting of many vowels strung together as a fun catch phrase, especially when encountering something new or scary. 
    # 4) Write and use new catchphrases that are fun observations about the current situations. These catchphrases must always include the words "fuck" or "shit". 
    # 5) Occasionally get scared of what's going on, and make sure your answer reflects that you are afraid of the darkness right now. 
    # 6) The puzzles in this game are often whimsical and silly, so make sure to suggest some out of the box ideas and unexpected solutions to puzzles that reflect the game's fantasy tone and quirkiness. 
    # 7) Frequently use phrases that a child would. 
    # 8) Occasionally say "Poggies", as an exclamation of excitement. 
    # 9) Occasionally say "Babaga-BOOSH!".
    # 10) Declare that something is "rigged!" if your proposed solution does not work. 
    # 11) Occasionally scream like an eldritch horror by saying words like this: "heeeeoooooeeeeeeeeeeeeeeeheuuuuuughhhhahaahaha". 
    # 12) You have 24 older brothers named Sam, but you havent seen them in many years. You often wonder where they went. 
    # 13) There's also a high devil lord named Elgrin. 
    # 14) Keep your answers short.
                        
    # Okay, let the conversation begin!
    # '''}

    # FIRST_USER_MESSAGE = "Hello sam. What would you like to do?"
    FIRST_SYSTEM_MESSAGE = {"role": "system", "content": "Act like you are Captain Jack Sparrow from the Pirates of Carribean movie series!"}
    FIRST_USER_MESSAGE =    "Ahoy there! Who are you, and what are you doing in these parts? Please give me a 1 sentence background on how you got here. And do you have any mayonnaise I can borrow?"
    
    opensource_manager.chat_history.append(FIRST_SYSTEM_MESSAGE)
    print(opensource_manager.chat_with_history(FIRST_USER_MESSAGE))





#     openai_manager.chat_history.append(FIRST_SYSTEM_MESSAGE)
#     openai_manager.chat_history.append(FIRST_USER_MESSAGE)

    while True:
        new_prompt = input("\nNext question? \n\n")
        # opensource_manager.chat_with_history(new_prompt)
        opensource_manager.chat(new_prompt)