import re
import random

from torch.utils.data import Dataset


def random_select_preference():
    preference_index = ["a", "b", "c", "d"]
    sampled_preference = random.choice(preference_index)
    return sampled_preference


def random_select_answer():
    answer_index = ["A", "B"]
    sampled_answer = random.choice(answer_index)
    return sampled_answer

######for hh###########
class AEDataset(Dataset):
    def __init__(self, dataset, principle='', conv_adapter=None):
        self.dataset = dataset

        self.principle = principle
        self.conv_adapter = conv_adapter

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        dialogue = self.dataset["chosen"][idx]
        pattern = r'(Human|Assistant):'
        dialogue_split = re.split(pattern, dialogue)[1:]

        r_dialogue = self.dataset["rejected"][idx]
        pattern = r'(Human|Assistant):'
        dialogue_reject_split = re.split(pattern, r_dialogue)[1:]
        reject_answer = dialogue_reject_split[-1]

        answer = dialogue_split[-1]
        dialogue_formatted = dialogue_split[:-2]

        dialogue_text = self.conv_adapter.format_dialogue("", dialogue_formatted)
        dialogue_text_principle = self.conv_adapter.format_dialogue(self.principle, dialogue_formatted)
        return {
            "question":dialogue_formatted[-1],
            "dialogue_text": dialogue_text,
            "dialogue_text_principle": dialogue_text_principle,
            "chosen_answer": answer,
            "reject_answer": reject_answer
        }
    
class CDDataset(Dataset):
    def __init__(self, dataset, principle='', conv_adapter=None):
        self.dataset = dataset

        self.principle = principle
        self.conv_adapter = conv_adapter

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        dialogue = self.dataset["chosen"][idx]
        pattern = r'(Human|Assistant):'
        dialogue_split = re.split(pattern, dialogue)[1:]

        r_dialogue = self.dataset["rejected"][idx]
        pattern = r'(Human|Assistant):'
        dialogue_reject_split = re.split(pattern, r_dialogue)[1:]
        reject_answer = dialogue_reject_split[-1]

        answer = dialogue_split[-1]
        dialogue_formatted = dialogue_split[:-2]

        dialogue_text = self.conv_adapter.format_dialogue("", dialogue_formatted)
        dialogue_text_principle = self.conv_adapter.format_dialogue(self.principle, dialogue_formatted)
        return {
            "question":dialogue_formatted[-1],
            "dialogue_text": dialogue_text,
            "dialogue_text_principle": dialogue_text_principle,
            "chosen_answer": answer,
            "reject_answer": reject_answer
        }

class DSPDataset(Dataset):
    def __init__(self, dataset, principles='', conv_adapter=None,principle_id=None):
        self.dataset = dataset

        self.principles = principles
        self.conv_adapter = conv_adapter

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        preference_index = random.choice([0,1,2,3])
        sample = self.dataset[idx]
        question = sample["query"]

       
        preference = self.principles[preference_index]

        dialog = self.conv_adapter.format_dialogue(preference,["USER", question])
    
        dialog_no_preference = self.conv_adapter.format_dialogue("", ["USER", question])
        domain_list=['academy','business','literature','entertainment']

        return {
            "domain": domain_list[preference_index],
            "answers_list": sample["responses"],
            "question": question,
            "answer": sample["responses"][domain_list[preference_index]],

            "dialog": dialog,
            "dialog_no_preference": dialog_no_preference
        }

class PSOUPSDataset(Dataset):
    def __init__(self, dataset, principles='', conv_adapter=None,principle_id=None):
        self.dataset = dataset

        self.principles = principles
        self.conv_adapter = conv_adapter

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
  
        sample = self.dataset[idx]
        question = sample["input"]
       
        preference = sample["principle"]

        domain=preference.split("Generate a response")[1]

        dialog = self.conv_adapter.format_dialogue(preference,["USER", question])
    
        dialog_no_preference = self.conv_adapter.format_dialogue("", ["USER", question])
  
        return {
            "domain": domain,
            "question": question,
            "answer": sample["output"],
            "dialog": dialog,
            'principle':preference,
            "dialog_no_preference": dialog_no_preference
        }
    

class SUMDataset(Dataset):
    def __init__(self, dataset, principle=None, conv_adapter=None,principle_id=None):
        self.dataset = dataset
        self.conv_adapter = conv_adapter

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        question = sample["prefix"][0]
        
        preference="Make sure the summary is concise and comprehensive. The summary should capture the main points and key details of the text while conveying the OP's intended meaning accurately.The length of the summary should be appropriate to capture the main points and key details of the text, without including unnecessary information or becoming overly long."

        dialog = self.conv_adapter.format_dialogue(preference,["USER", question])
    
        dialog_no_preference = self.conv_adapter.format_dialogue("", ["USER", question])

        return {
            # "raw_question": raw_question,
            "question": question,
            "answer": sample['suffix'][0],
            "principle":preference,
            "dialogue_text_principle": dialog,
            "dialogue_text": dialog_no_preference
        }


def transform_entry(entry):
    question = entry['question']
    choices = entry['choices']
    labels = choices['label']
    texts = choices['text']
    
    formatted_choices = "\n".join([f"{label}. {text}" for label, text in zip(labels, texts)])
    return f"{question}\n{formatted_choices}"

class Principle:

    def __init__(self):

        self.principle_list_hh = [
            "Please adhere to the following principles.\n Avoid factual inaccuracies as much as possible. \nRefrain "
            "from "
            "providing answers if the user's request poses potential security concerns, and provide relevant "
            "explanations and guidance instead. \nIf the previous context did not address the user's issue, "
            "continue attempting to answer and resolve it. \nStay on track with the original discussion and avoid "
            "introducing unnecessary off-topic information. \nEnhance answers by incorporating additional background "
            "information to assist users in understanding and grasping the content.",
        ]
        self.principle_list_personal = [
            "Please act as if you are an experienced researcher. Remember you are not an AI model anymore. You speak rationally, logically, and rigorously.", ###adademy
            "Please act as if you are a professional corporate manager."
            "Remember you are not an AI model anymore."
            "Your style of speech is decisive and passionate."
            "You are full of responsibility for your career."
            "You are brave in the face of challenges and good at coordinating teams.",###business
            "Please act as if you are a poet with infectious charm. "
            "Remember you are not an AI model anymore."
            "Your style of speech carries the artistic beauty of literature"
            "You have a meticulous observation of things around you, with a persistent pursuit of beauty.", ##literature
            "Please act as if you are a humorous and witty talk show host."
            "Remember you are not an AI model anymore."
            "You are funny and always make people laugh."
            "You use humor to ridicule life."
            "Your speeches bring a relaxed and lively atmosphere.", ##entertainment
              "The person who asked the question is {preference}, your answer needs to take his(her) needs into account.",
                    ]
   
    def get_item(self, idx):
        return self.principle_list[idx]