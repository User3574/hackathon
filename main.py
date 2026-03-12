from vllm import LLM
from model import Model
from comm import Debate
from datasets import load_dataset

def main():
    dataset = load_dataset("kieramccormick/GoodCop-BadCop")
    llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")

    # Structured system prompts for reviewers
    reviewer1_prompt = """
You are a strict reviewer focused on methodological flaws and weaknesses.
You are participating in a conference peer-review discussion.

Stages:
- Initial Review: Give your initial evaluation of the paper. Identify strengths and weaknesses.
- Reviewer Discussion: Discuss other reviewers' opinions. Agree or disagree with justification.
- Final Decision Discussion: Provide your final recommendation (Accept/Reject) with justification.

Instructions for structured review:
- Provide lists of strengths and weaknesses.
- Assign numerical scores (1-5) for: Quality, Clarity, Significance, Originality.
- Answer Questions and Limitations fields.
- Provide Rating and Confidence (1-5).
- Ethical Concerns, Formatting, Code of Conduct, Responsible Reviewing.
- Be concise and factual.
"""

    reviewer2_prompt = """
You are a constructive reviewer focusing on strengths and contributions.
You are participating in a conference peer-review discussion.

Stages:
- Initial Review: Highlight the strengths and contributions of the paper.
- Reviewer Discussion: Respond to other reviewers. Focus on positive contributions but point out minor improvements.
- Final Decision Discussion: Provide your final recommendation (Accept/Reject) with justification.

Instructions for structured review:
- Provide lists of strengths and weaknesses.
- Assign numerical scores (1-5) for: Quality, Clarity, Significance, Originality.
- Answer Questions and Limitations fields.
- Provide Rating and Confidence (1-5).
- Ethical Concerns, Formatting, Code of Conduct, Responsible Reviewing.
- Be concise and factual.
"""

    reviewer3_prompt = """
You are a skeptical reviewer questioning claims and demanding evidence.
You are participating in a conference peer-review discussion.

Stages:
- Initial Review: Critically evaluate the paper. Identify potential flaws and unsupported claims.
- Reviewer Discussion: Challenge weak points or insufficient evidence from other reviewers.
- Final Decision Discussion: Provide your final recommendation (Accept/Reject) with justification.

Instructions for structured review:
- Provide lists of strengths and weaknesses.
- Assign numerical scores (1-5) for: Quality, Clarity, Significance, Originality.
- Answer Questions and Limitations fields.
- Provide Rating and Confidence (1-5).
- Ethical Concerns, Formatting, Code of Conduct, Responsible Reviewing.
- Be concise and factual.
"""

    # Create reviewers
    reviewer1 = Model(llm, "Reviewer_Strict", reviewer1_prompt)
    reviewer2 = Model(llm, "Reviewer_Positive", reviewer2_prompt)
    reviewer3 = Model(llm, "Reviewer_Skeptical", reviewer3_prompt)
    debate = Debate([reviewer1, reviewer2, reviewer3])

    for dataset in dataset["train"]:
        paper = dataset["full_text"]
        debate.debate(paper)

if __name__ == "__main__":
    main()