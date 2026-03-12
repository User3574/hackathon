import json

class Debate:
    def __init__(self, models, output_file="review_log"):
        self.id = 0
        self.models = models
        self.output_file = output_file
        self.rounds = [
            "Initial Review",
            "Reviewer Discussion",
            "Final Decision Discussion"
        ]
        self.summarizer_prompt = """
You are a meta-reviewer. Your task is to summarize the discussion of the reviewers above.
Produce a structured meta-review in the following format:

Summary:
Strengths:
Weaknesses:
Quality: (1-5)
Clarity: (1-5)
Significance: (1-5)
Originality: (1-5)
Questions:
Limitations:
Rating: (1-5 and textual, e.g., 4: Borderline Accept)
Confidence: (1-5)
Ethical Concerns: 
Paper Formatting Concerns:
Code Of Conduct Acknowledgement:
Responsible Reviewing Acknowledgement:
Final Justification:
"""
        # Optional summarizer agent
        self.summarizer = models[0]

    def save(self, chat):
        with open(f"{self.output_file}_{self.id}.json", "w") as f:
            json.dump(chat, f, indent=2)
        self.id += 1

    def debate(self, paper_summary):
        chat = [{
            "role": "paper",
            "content": paper_summary
        }]

        for round_name in self.rounds:
            print(f"\n===== {round_name} =====\n")
            for model in self.models:
                print(f"-------- {model.name} is responding... --------")
                chat = model.comm(chat, round_name)
                print(f"{chat[-1]['content']}\n")
                print(f"-----------------------\n")

        # Generate a final summary/conclusion using the summarizer
        chat.append({
            "role": "system",
            "content": self.summarizer_prompt
        })
        chat = self.summarizer.comm(chat, "Final Summary")

        print(f"\n===== Final Summary =====\n")
        print(chat[-1]['content'], "\n")
        self.save(chat)
        return chat