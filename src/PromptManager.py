import os
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PromptManager:
    def __init__(self, file_path):
        self.file_path = file_path
        self.create_json_file_if_not_exists()
        self.prompts = self.load()

    def create_json_file_if_not_exists(self):
        if not os.path.exists(self.file_path):
            try:
                with open(self.file_path, 'w', encoding='utf-8') as f:
                    json.dump({}, f, ensure_ascii=False, indent=4)
                logging.info(f"Created an empty JSON file at {self.file_path}.")
            except IOError as e:
                logging.error(f"Failed to create JSON file: {e}")

    def load(self):
        if not os.path.exists(self.file_path):
            logging.info(f"File {self.file_path} does not exist. Starting with an empty dictionary.")
            return {}
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if not isinstance(data, dict):
                    logging.error("The file does not contain a valid JSON object.")
                    return {}
                return data
        except json.JSONDecodeError as e:
            logging.error(f"Failed to load JSON file: {e}")
            return {}

    def add(self, key, value):
        if not isinstance(key, str):
            raise TypeError("Key must be a string.")
        if key in self.prompts:
            raise KeyError(f"Prompt with key '{key}' already exists.")
        self.prompts[key] = value
        self.save()
        logging.info(f"Prompt '{key}' has been added.")

    def remove(self, key):
        if key not in self.prompts:
            logging.info(f"Prompt with key '{key}' does not exist.")
            return
        confirm = input(f"Are you sure you want to remove the '{key}' prompt? (y/n): ")
        if confirm.lower() == 'y':
            del self.prompts[key]
            self.save()
            logging.info(f"Prompt '{key}' has been removed.")
        else:
            logging.info(f"Deletion of prompt '{key}' canceled.")

    def update(self, key, value):
        if key not in self.prompts:
            raise KeyError(f"Prompt with key '{key}' does not exist.")
        self.prompts[key] = value
        self.save()
        logging.info(f"Prompt '{key}' has been updated.")

    def get(self, key):
        prompt = self.prompts.get(key, 'No this prompt')
        if prompt == 'No this prompt':
            logging.warning(f"Prompt with key '{key}' does not exist.")
            prompt = self.prompts.get("panda")
        return prompt

    def show_keys(self):
        keys = list(self.prompts.keys())
        if not keys:
            logging.info("No prompts available.")
        else:
            logging.info("Available prompts:")
            for key in keys:
                logging.info(f"* {key}")

    def save(self):
        try:
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump(self.prompts, f, ensure_ascii=False, indent=4)
            logging.info(f"Prompts have been saved to {self.file_path}.")
        except IOError as e:
            logging.error(f"Failed to save prompts: {e}")

if __name__ == "__main__":
    prompt_manager = PromptManager("../prompts.json")
    prompt_manager.show_keys()
    #print(prompt_manager.get("panda"))
    # prompt_manager.add(key = "", value = "")
