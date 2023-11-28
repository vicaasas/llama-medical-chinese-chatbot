import os, json, itertools, bisect, gc

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import transformers
import torch
from accelerate import Accelerator
import accelerate
import time
import opencc
# from peft import (  # noqa: E402
#     LoraConfig,
#     # BottleneckConfig,
#     get_peft_model,
#     get_peft_model_state_dict,
#     prepare_model_for_int8_training,
#     set_peft_model_state_dict,
# )
from peft import PeftModel
from transformers import GenerationConfig
CONVERTER_T2S = opencc.OpenCC("t2s.json")
CONVERTER_S2T = opencc.OpenCC("s2t.json")
model = None
tokenizer = None
generator = None
# os.environ["CUDA_VISIBLE_DEVICES"]="1"

def load_model(model_name, eight_bit=0, device_map="auto"):
    global model, tokenizer, generator

    print("Loading "+model_name+"...")

    if device_map == "zero":
        device_map = "balanced_low_0"

    # config
    gpu_count = torch.cuda.device_count()
    print('gpu_count', gpu_count)

    tokenizer = transformers.LlamaTokenizer.from_pretrained(model_name)
    # model_llama
    model = transformers.LlamaForCausalLM.from_pretrained(
        model_name,
        #device_map=device_map,
        # device_map="auto",
        # torch_dtype=torch.float16,
        # max_memory = {0: "14GB", 1: "14GB", 2: "14GB", 3: "14GB",4: "14GB",5: "14GB",6: "14GB",7: "14GB"},
        # load_in_8bit=True,
        #from_tf=True,
        # low_cpu_mem_usage=True,
        # load_in_8bit=False,
        # cache_dir="cache"
    )
    # 使用lora 請將下面解註 而上面的 model_name 請用"FlagAlpha/Llama2-Chinese-7b-Chat" 並將 model 變數改成 model_llama
    # print(f"using lora vickt/LLama-chinese-med-chat-lora")
    # model = PeftModel.from_pretrained(
    #     model_llama,
    #     "vickt/LLama-chinese-med-chat-lora",
    #     torch_dtype=torch.float16,
    #     )
    model.to("cuda:1")
    model.eval()
    # generator = model.generate

# load_model("./pretrained/zh_tune")
# load_model("./pretrained/en_tune")
# load_model("./pretrained/en_tune_100")
load_model("vickt/LLama-chinese-med-chat")
# load_model("daryl149/llama-2-7b-chat-hf")
# load_model("FlagAlpha/Llama2-Chinese-7b-Chat")
# load_model("TheBloke/Llama-2-13B-fp16")

# First_chat = "ChatDoctor: I am ChatDoctor, what medical questions do you have?"
First_chat = "AI Doctor: 您好，我是 AI Doctor, 請問需要甚麼幫忙"
print(First_chat)
history = []
# history.append(First_chat)
# generation_config = GenerationConfig(
#     temperature=0.5,
#     top_p=1.0,
#     top_k=50,
#     num_beams=4,
# )
def go():
    invitation = "AI Doctor: "
    human_invitation = "user: "

    # input
    msg = input(human_invitation)
    print("")

    history.append(human_invitation + msg)
    # fulltext = "如果您是醫生，請根據患者的描述回答醫學問題。 \n\n" + "\n\n".join(history) + "\n\n" + invitation

    fulltext = "如果您是醫生，請根據患者的描述回答醫學問題。 \n\n" + human_invitation + msg + "\n\n" + invitation
    # fulltext = "If you are a doctor, please answer the medical questions based on the patient's description. \n\n" + "\n\n".join(history) + "\n\n" + invitation
    # fulltext = "\n\n".join(history) + "\n\n" + invitation
    
    #print('SENDING==========')
    #print(fulltext)
    #print('==========')

    generated_text = ""
    gen_in = tokenizer(CONVERTER_T2S.convert(fulltext), return_tensors="pt").input_ids.to("cuda:1")
    # gen_in = tokenizer(fulltext, return_tensors="pt").input_ids.cuda()
    in_tokens = len(gen_in)
    with torch.no_grad():
            generated_ids = model.generate(
                input_ids=gen_in,
                # generation_config=generation_config,
                max_new_tokens=400,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
                num_return_sequences=1,
                do_sample=True,
                repetition_penalty=1.1, # 1.0 means 'off'. unfortunately if we penalize it it will not output Sphynx:
                temperature=0.5, # default: 1.0
                top_k = 50, # default: 50
                # top_p = 1.0, # default: 1.0
                early_stopping=True,
            )
            generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0] # for some reason, batch_decode returns an array of one element?

            text_without_prompt = generated_text[len(fulltext):]

    response = text_without_prompt

    response = response.split(human_invitation)[0]

    response.strip()

    print(CONVERTER_S2T.convert(invitation + response))

    print("")

    history.append(invitation + response)

while True:
    go()
