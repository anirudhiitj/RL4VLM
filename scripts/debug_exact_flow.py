"""
Exact replica of the training flow to debug action_tokens_log_prob.
Uses the same code paths as main.py.
"""
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
sys.path.insert(0, "/mnt/raid/rl_gaming/RL4VLM/VLM_PPO")

# Same as main.py line 1-2
from patch import replace_llama_attn_with_xformers_attn
replace_llama_attn_with_xformers_attn()

import torch
import gymnasium as gym
import gym_cards
from functools import partial
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer

from a2c_ppo_acktr.llava_interface import init_pretrained_model, find_all_linear_names
from a2c_ppo_acktr.llava_interface import llava_generate, llava_evaluate
from a2c_ppo_acktr.model import VLMPolicy, VLMValue
from a2c_ppo_acktr.rl_utils import get_prompt, text_projection

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token
from llava.model.language_model.llava_mistral import LlavaMistralForCausalLM

import argparse

# Minimal args object mimicking the training args
class Args:
    temperature = 0.2
    num_beams = 1
    max_new_tokens = 256
    thought_prob_coef = 0.5
    conv_mode = "llava_v1"  # DEFAULT in arguments.py
    env_name = "gym_cards/NumberLine-v0"
    action_only_prompt = False

args = Args()

print("Loading model...")
model_path = "/mnt/raid/rl_gaming/RL4VLM/checkpoints/sft_numberline"
tokenizer = AutoTokenizer.from_pretrained(model_path)

if 'mistral' in model_path.lower():
    base = LlavaMistralForCausalLM.from_pretrained(model_path)
else:
    from llava.model import LlavaLlamaForCausalLM
    base = LlavaLlamaForCausalLM.from_pretrained(model_path)

base, tokenizer = init_pretrained_model(base, tokenizer, pretrain_mm_adapter=None)
image_processor = base.get_vision_tower().image_processor

# Apply LoRA like main.py does
base_lora_config = LoraConfig(
    r=128, lora_alpha=256,
    target_modules=find_all_linear_names(base, "all"),
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
)
base = get_peft_model(base, base_lora_config)
value_model = VLMValue(base)
value_model = value_model.cuda()

print("Model loaded. Creating environment...")
env = gym.make("gym_cards/NumberLine-v0")
obs_raw, _ = env.reset(seed=1)
# Wrap obs like make_vec_envs does
obs = torch.from_numpy(obs_raw).unsqueeze(0)  # [1, H, W, C]

# Build prompt exactly like main.py
infos = None
qs = get_prompt(args.env_name, args.action_only_prompt, infos)
qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
conv = conv_templates[args.conv_mode].copy()
conv.append_message(conv.roles[0], qs)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()

print("\n=== PROMPT ===")
print(repr(prompt[:200]))
print("=== conv_mode:", args.conv_mode, "===\n")

INPUT_IDS = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)
INPUT_IDS[INPUT_IDS == 0] = 259

print("INPUT_IDS shape:", INPUT_IDS.shape)
print("INPUT_IDS (first 20):", INPUT_IDS[0, :20].tolist())

# Process observation like VLMPolicy.process_obs
processed_images = obs
image_tensor = image_processor.preprocess(processed_images, return_tensors='pt')['pixel_values'].to(dtype=base.dtype)

print("\nRunning llava_generate (same as actor_critic.act)...")
with torch.no_grad():
    values, padded_output_ids, text_outputs, sum_log_probs, action_tokens_log_prob = llava_generate(
        value_model=value_model,
        tokenizer=tokenizer,
        input_ids=INPUT_IDS,
        image_tensor=image_tensor,
        args=args
    )

print("\n" + "="*60)
print("RESULTS:")
print("="*60)
print("Generated text:", text_outputs[0] if text_outputs else "EMPTY")
print()
print("padded_output_ids shape:", padded_output_ids.shape)
nonzero = padded_output_ids[0][padded_output_ids[0] != 0].tolist()
print("Non-zero tokens ({} total):".format(len(nonzero)))
print(nonzero)
print()
print("values:", values)
print("sum_log_probs:", sum_log_probs)
print("action_tokens_log_prob:", action_tokens_log_prob)
print()

if action_tokens_log_prob.abs().max().item() < 1e-6:
    print("!!! action_tokens_log_prob IS ZERO — BUG STILL PRESENT !!!")
    print()
    
    # Deep debug — manually run the matching logic
    print("--- Manual debug of matching logic ---")
    output_ids = padded_output_ids.to(value_model.base.device)
    unfolded = output_ids.unfold(dimension=-1, size=3, step=1)
    
    target1 = torch.tensor([345,1774,1264]).to(output_ids.device)
    target2 = torch.tensor([28739,1774,1264]).to(output_ids.device)
    matches1 = (unfolded == target1).all(dim=-1)
    matches2 = (unfolded == target2).all(dim=-1)
    matches = matches1 | matches2
    
    print("target1 [345,1774,1264] matches:", matches1.any().item(), "positions:", matches1.nonzero(as_tuple=True)[-1].tolist())
    print("target2 [28739,1774,1264] matches:", matches2.any().item(), "positions:", matches2.nonzero(as_tuple=True)[-1].tolist())
    print("combined matches:", matches.any().item())
    
    match_index = matches.nonzero(as_tuple=True)[-1]
    print("match_index:", match_index)
    
    if match_index.shape[0] >= 1:
        mi = match_index[-1].unsqueeze(0)
        print("Using match_index:", mi)
    else:
        output_ids_mask = (output_ids != 0)[:, 1:]
        mi = output_ids_mask.nonzero(as_tuple=False)[-4,1]
        print("FALLBACK match_index:", mi)
    
    # Check all 3-grams with token 1774
    id_list = output_ids[0].tolist()
    print("\nAll positions with token 1774:")
    for idx, tid in enumerate(id_list):
        if tid == 1774:
            start = max(0, idx-3)
            end = min(len(id_list), idx+4)
            context = id_list[start:end]
            decoded = [tokenizer.decode([t]) for t in context if t != 0]
            print("  pos {}: {} -> {}".format(idx, context, decoded))
    
    # Check ALL trigrams for any pattern containing "action"
    print("\nSearching all trigrams for 'action' (token 1774)...")
    for i in range(unfolded.size(1)):
        tri = unfolded[0, i].tolist()
        if 1774 in tri:
            print("  pos {}: {} -> '{}'".format(i, tri, tokenizer.decode(tri)))
else:
    print(">>> action_tokens_log_prob is NON-ZERO: {} — FIX WORKS!".format(action_tokens_log_prob))

# Also check the debug file
print("\n--- Debug file output ---")
if os.path.exists("/tmp/rl_token_debug.txt"):
    with open("/tmp/rl_token_debug.txt") as f:
        print(f.read())
else:
    print("No debug file found")

print("\nDONE")
