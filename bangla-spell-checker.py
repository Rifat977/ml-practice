import pandas as pd
import difflib

class BanglaSpellChecker:
    def __init__(self):
        self.words = []
    
    def load_data(self, file_paths):
        for file_path in file_paths:
            data = pd.read_csv(file_path, header=None)[0].values
            self.words.extend(data)
        
    def train(self):
        pass  # No training is required for this approach
    
    def suggest_correction(self, input_text):
        words = input_text.split()
        corrections = []
        
        for word in words:
            if word not in self.words:
                word_corrections = difflib.get_close_matches(word, self.words, n=1, cutoff=0.8)
                corrections.append(word_corrections)
        
        return corrections
    
    def suggest_completion(self, input_text):
        words = input_text.split()
        last_word = words[-1]

        if last_word:
            prefix = " ".join(words[:-1])
            completions = [prefix + " " + suggestion for suggestion in self.words if suggestion.startswith(last_word)]
            return completions
        else:
            return []
    
spell_checker = BanglaSpellChecker()
spell_checker.load_data("data/bangla_word_huge_dataset.csv")
spell_checker.train()

while True:
    user_input = input("Enter some text: ")
    completions = spell_checker.suggest_completion(user_input)
    suggestions = spell_checker.suggest_correction(user_input)
    
    print("Completions:")
    for completion in completions:
        print(completion)
    
    print("Suggestions:", suggestions)
