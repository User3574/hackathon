from vllm import SamplingParams


class Model:
    def __init__(self, llm, name, system_prompt):
        self.llm = llm
        self.name = name
        self.system_prompt = system_prompt

        self.sampling_params = SamplingParams(
            temperature=0.6,
            top_p=0.9,
            max_tokens=1024
        )

    def build_prompt(self, chat, round_name):
        prompt = f"system: {self.system_prompt}\n"
        prompt += f"Current review stage: {round_name}\n\n"
        for msg in chat:
            prompt += f"{msg['role']}: {msg['content']}\n"
        prompt += f"{self.name}:"
        return prompt

    def comm(self, chat, round_name):
        prompt = self.build_prompt(chat, round_name)
        outputs = self.llm.generate([prompt], self.sampling_params)
        response = outputs[0].outputs[0].text.strip()
        chat.append({
            "role": self.name,
            "content": response
        })
        return chat