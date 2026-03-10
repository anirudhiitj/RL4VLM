"""Debug: replicate EXACT training flow with LoRA + llava_v1 conv_mode."""
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
sys.path.insert(0, "/mnt/raid/rl_gaming/RL4VLM/VLM_PPO")

from patch import replace_llama_attn_with_xformers_attn
replace_llama_attn_with_xformers_attn()

import torch
import gymnasium as gym
import gym_cards
from functools import partial
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer
from llava.model.language_model.llava_mistral import LlavaMistralForCausalLM
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token
from a2c_ppo_acktr.llava_interface import init_pretrained_model, find_all_linear_names
from a2c_ppo_acktr.llava_interface import llava_generate, llava_evaluate
from a2c_ppo_acktr.model import VLMValue
from a2c_ppo_acktr.rl_utils import text_projection
import argparse

model_path = "/mnt/raid/rl_gaming/RL4VLM/checkpoints/sft_numberline"

# Check: does 'mistral' appear in the path?
print("'mistral' in model_path:", 'mistral' in model_path.lower())
# This means the training uses LlavaLlamaForCausalLM, NOT Mistral!

# Load EXACTLY like main.py does
tokenizer = AutoTokenizer.from_pretrained(model_path)
# main.py checks 'mistral' in model_path.lower() — it's NOT there
# so it loads LlavaLlamaForCausalLM
print("Loading with LlavaLlamaForCausalLM (same as main.py)...")
base = LlavaLlamaForCausalLM.from_pretrained(model_path)

base.config.max_length = 1024
base, tokenizer = init_pretrained_model(base, tokenizer, pretrain_mm_adapter=None)
image_processor = base.get_vision_tower().image_processor

# Apply LoRA like main.py
base_lora_config = LoraConfig(
    r=128, lora_alpha=256,
    target_modules=find_all_linear_names(base, "all"),
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
base = get_peft_model(base, base_lora_config)
print("LoRA applied")

value_model = VLMValue(base)
value_model = value_model.cuda()

# Create env
env = gym.make("gym_cards/NumberLine-v0")
obs_raw, _ = env.reset(seed=1)
# main.py uses make_vec_envs which wraps obs differently
# The obs from vec_env is a tensor of shape [1, H, W, C]
obs = torch.from_numpy(obs_raw).unsqueeze(0)

# Build prompt with llava_v1 conv_mode (same as main.py default)
qs = """You are playing a game called number line. You will see a target number and a current number in the image. And your goal is to move the current number closer to the target by choosing either adding or subtracting one to the current number. Your response should be a valid json file in the following format:
{
 "current number": "x",
"target number": "x",
"thoughts": "{first read out the current and target number, then think carefully about which action to choose}",
"action": "-" or "+"
}"""
qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

# Use llava_v1 (training default), NOT mistral_instruct
conv = conv_templates["llava_v1"].copy()
conv.append_message(conv.roles[0], qs)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()
print("\nPrompt format (first 200 chars):", repr(prompt[:200]))

INPUT_IDS = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)
INPUT_IDS[INPUT_IDS == 0] = 259
print("INPUT_IDS shape:", INPUT_IDS.shape)

# Create args namespace like main.py
args = argparse.Namespace(
    temperature=0.2, num_beams=1, max_new_tokens=256,
    thought_prob_coef=0.5)

# Process image like VLMPolicy.process_obs does
image_tensor = image_processor.preprocess(obs, return_tensors='pt')['pixel_values'].to(dtype=base.dtype)
print("image_tensor shape:", image_tensor.shape, "dtype:", image_tensor.dtype)

# Call llava_generate (same as main.py)
print("\nCalling llava_generate...")
values, padded_output_ids, text_outputs, sum_log_probs, action_tokens_log_prob = llava_generate(
    value_model=value_model, tokenizer=tokenizer, input_ids=INPUT_IDS,
    image_tensor=image_tensor, args=args)

print("\n" + "="*60)
print("text_outputs:", text_outputs)
print("sum_log_probs:", sum_log_probs)
print("action_tokens_log_prob:", action_tokens_log_prob)
print("="*60)

# Check the token IDs
nonzero = padded_output_ids[0][padded_output_ids[0] != 0].tolist()
print("\nOutput token IDs ({} tokens): {}".format(len(nonzero), nonzero))

# Pattern matching analysis
unfolded = padded_output_ids.unfold(dimension=-1, size=3, step=1)
target1 = torch.tensor([345,1774,1264]).to(padded_output_ids.device)
target2 = torch.tensor([28739,1774,1264]).to(padded_output_ids.device)
matches1 = (unfolded == target1).all(dim=-1)
matches2 = (unfolded == target2).all(dim=-1)
print("\ntarget1 [345,1774,1264] matches:", matches1.any().item(), matches1.nonzero(as_tuple=True)[-1].tolist())
print("target2 [28739,1774,1264] matches:", matches2.any().item(), matches2.nonzero(as_tuple=True)[-1].tolist())

# Check for 1774 context
id_list = padded_output_ids[0].tolist()
for idx, tid in enumerate(id_list):
    if tid == 1774:
        start = max(0, idx-3)
        end = min(len(id_list), idx+4)
        decoded = [(id_list[i], tokenizer.decode([id_list[i]])) for i in range(start, end) if id_list[i] != 0]
        print("token 1774 at pos {}, context: {} decoded: {}".format(idx, id_list[start:end], decoded))

if action_tokens_log_prob.abs().max().item() < 1e-6:
    print("\n>>> PROBLEM: action_tokens_log_prob is ZERO!")
else:
    print("\n>>> OK: action_tokens_log_prob is non-zero: {}".format(action_tokens_log_prob))

# Also check the debug file
if os.path.exists("/tmp/rl_token_debug.txt"):
    print("\n--- Debug file contents ---")
    with open("/tmp/rl_token_debug.txt") as f:
        print(f.read())

print("\nDONE")
