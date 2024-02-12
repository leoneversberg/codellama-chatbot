import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


class ChatModel:
    def __init__(self, model="codellama/CodeLlama-7b-Instruct-hf"):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            quantization_config=quantization_config,
            device_map="cuda",
            cache_dir="./models",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model, use_fast=True, padding_side="left"
        )

        self.history = []
        self.history_length = 1

        self.DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant with a deep knowledge of code and software design. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\
        """

    def generate(
        self, user_prompt, system_prompt, top_p=0.9, temperature=0.1, max_new_tokens=512
    ):

        texts = [f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n"]
        # The first user input is _not_ stripped
        do_strip = False
        for old_prompt, old_response in self.history:
            old_prompt = old_prompt.strip() if do_strip else old_prompt
            do_strip = True
            texts.append(f"{old_prompt} [/INST] {old_response.strip()} </s><s>[INST] ")
        user_prompt = user_prompt.strip() if do_strip else user_prompt
        texts.append(f"{user_prompt} [/INST]")
        prompt = "".join(texts)

        inputs = self.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False
        ).to("cuda")

        output = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pad_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=top_p,
            top_k=50,
            temperature=temperature,
        )
        output = output[0].to("cpu")
        response = self.tokenizer.decode(output[inputs["input_ids"].shape[1] : -1])
        self.append_to_history(user_prompt, response)
        return response

    def append_to_history(self, user_prompt, response):
        self.history.append((user_prompt, response))
        if len(self.history) > self.history_length:
            self.history.pop(0)
