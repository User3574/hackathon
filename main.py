from model import Model
from comm import Debate

model1 = Model(
    "Qwen/Qwen2.5-1.5B-Instruct",
    system_prompt="You are a very skeptical AI that challenges weak arguments.",
    name="Skeptic"
)
model2 = Model(
    "Qwen/Qwen2.5-1.5B-Instruct",
    system_prompt="You are an optimistic AI that supports new ideas.",
    name="Optimist"
)

debate = Debate([model1, model2])
debate.debate(
    "Is open source AI better than closed source AI?",
    rounds=3
)
