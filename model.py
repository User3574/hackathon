from vllm import LLM, SamplingParams


class Model:
    def __init__(self, model_path, system_prompt="You are a helpful assistant.", name=None):
        self.model_path = model_path
        self.system_prompt = system_prompt
        self.name = name if name else model_path

        self.llm = LLM(
            model=self.model_path,
            tokenizer=self.model_path,
            trust_remote_code=False
        )

        self.sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=100
        )

    def build_prompt(self, chat):
        prompt = f"system: {self.system_prompt}\n"

        for msg in chat:
            prompt += f"{msg['role']}: {msg['content']}\n"

        prompt += f"{self.name}:"
        return prompt

    def comm(self, chat):
        prompt = self.build_prompt(chat)

        outputs = self.llm.generate([prompt], self.sampling_params)
        response = outputs[0].outputs[0].text.strip()

        chat.append({
            "role": self.name,
            "content": response
        })

        return chat
