import json


class Debate:
    def __init__(self, models, output_file="debate_log.json"):
        self.models = models
        self.output_file = output_file

    def save(self, chat):
        with open(self.output_file, "w") as f:
            json.dump(chat, f, indent=2)

    def debate(self, prompt, rounds=3):
        chat = []

        chat.append({
            "role": "user",
            "content": prompt
        })

        for r in range(rounds):
            print(f"\n--- Round {r+1} ---")

            for model in self.models:
                chat = model.comm(chat)

                print(f"{model.name}: {chat[-1]['content']}\n")

        self.save(chat)
        return chat
