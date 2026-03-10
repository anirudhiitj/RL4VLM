"""Debug script: Load model, run one inference, print exact token IDs."""
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
sys.path.insert(0, "/mnt/raid/rl_gaming/RL4VLM/VLM_PPO")

from patch import replace_llama_attn_with_xformers_attn
replace_llama_attn_with_xformers_attn()

import torch
import gymnasium as gym
import gym_cards
from transformers import AutoTokenizer
from llava.model.language_model.llava_mistral import LlavaMistralForCausalLM
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token
from a2c_ppo_acktr.llava_interface import init_pretrained_model

model_path = "/mnt/raid/rl_gaming/RL4VLM/checkpoints/sft_numberline"
tokenizer = AutoTokenizer.from_pretrained(model_path)
base = LlavaMistralForCausalLM.from_pretrained(model_path)
base, tokenizer = init_pretrained_model(base, tokenizer, pretrain_mm_adapter=None)
image_processor = base.get_vision_tower().image_processor
base = base.cuda()

# Create env and get observation
env = gym.make("gym_cards/NumberLine-v0")
obs, _ = env.reset(seed=1)

# Build prompt (same as main.py)
qs = """You are playing a game called number line. You will see a target number and a current number in the image. And your goal is to move the current number closer to the target by choosing either adding or subtracting one to the current number. Your response should be a valid json file in the following format:
{
 "current number": "x",
"target number": "x",
"thoughts": "{first read out the current and target number, then think carefully about which action to choose}",
"action": "-" or "+"
}"""
qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
conv = conv_templates["mistral_instruct"].copy()
conv.append_message(conv.roles[0], qs)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()

INPUT_IDS = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)
INPUT_IDS[INPUT_IDS == 0] = 259

# Process image
processed = image_processor.preprocess(obs, return_tensors='pt')['pixel_values'].to(dtype=base.dtype, device=base.device)

# Prepare multimodal inputs
_, _, _, _, inputs_embeds, _ = base.prepare_inputs_labels_for_multimodal(
    INPUT_IDS.to(base.device), None, None, None, None, processed)
inputs_embeds = inputs_embeds.to(base.device, dtype=base.dtype)

# Generate
with torch.inference_mode():
    outputs = base.generate(
        inputs_embeds=inputs_embeds,
        do_sample=True,
        temperature=0.2,
        num_beams=1,
        max_new_tokens=256,
        use_cache=True,
        output_scores=True,
        output_hidden_states=True,
        return_dict_in_generate=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    output_ids = outputs['sequences']

# Decode and print
text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
print("\n" + "="*60)
print("GENERATED TEXT:")
print(text)
print("="*60)

# Print ALL non-zero output_ids
nonzero = output_ids[0][output_ids[0] != 0].tolist()
print("\nALL output token IDs ({} tokens):".format(len(nonzero)))
print(nonzero)

# Now pad like main.py does
padded_output_ids = torch.zeros(output_ids.size(0), 2*256).to(dtype=output_ids.dtype, device=output_ids.device)
padded_output_ids[:, :output_ids.size(1)] = output_ids

print("\nPadded output_ids shape:", padded_output_ids.shape)
nonzero_padded = padded_output_ids[0][padded_output_ids[0] != 0].tolist()
print("Padded non-zero IDs ({} tokens):".format(len(nonzero_padded)))
print(nonzero_padded)

# Search for "action" related tokens
print("\n--- Searching for token 1774 (action) ---")
id_list = padded_output_ids[0].tolist()
for idx, tid in enumerate(id_list):
    if tid == 1774:
        start = max(0, idx-5)
        end = min(len(id_list), idx+6)
        print("Found 1774 at pos {}, context: {}".format(idx, id_list[start:end]))
        # Decode surrounding tokens individually
        for i in range(start, end):
            if id_list[i] != 0:
                print("  pos {} = {} -> '{}'".format(i, id_list[i], tokenizer.decode([id_list[i]])))

# Try all 3-grams containing 1774
print("\n--- All 3-grams containing 1774 ---")
unfolded = padded_output_ids.unfold(dimension=-1, size=3, step=1)
for i in range(unfolded.size(1)):
    trigram = unfolded[0, i].tolist()
    if 1774 in trigram:
        print("pos {}: {} -> '{}'".format(i, trigram, tokenizer.decode(trigram)))

# Check what "action": tokenizes to from this tokenizer
print("\n--- Tokenizer check ---")
for test_str in ['"action":', ' "action":', '\n"action":', '\\n"action":',
                 '"action"', 'action', '"action":']:
    ids = tokenizer.encode(test_str, add_special_tokens=False)
    print("'{}' -> {}".format(test_str.replace('\n','\\n'), ids))

# Try the matching from interface.py
print("\n--- Pattern matching test ---")
target1 = torch.tensor([345,1774,1264]).to(padded_output_ids.device)
target2 = torch.tensor([28739,1774,1264]).to(padded_output_ids.device)
matches1 = (unfolded == target1).all(dim=-1)
matches2 = (unfolded == target2).all(dim=-1)
matches = matches1 | matches2
print("target1 [345,1774,1264] matches:", matches1.any().item(), matches1.nonzero(as_tuple=True)[-1].tolist())
print("target2 [28739,1774,1264] matches:", matches2.any().item(), matches2.nonzero(as_tuple=True)[-1].tolist())
print("combined matches:", matches.any().item(), matches.nonzero(as_tuple=True)[-1].tolist())

# Also try with just [1774, 1264] (2-gram)
unfolded2 = padded_output_ids.unfold(dimension=-1, size=2, step=1)
target_2g = torch.tensor([1774,1264]).to(padded_output_ids.device)
matches_2g = (unfolded2 == target_2g).all(dim=-1)
print("\n2-gram [1774,1264] matches:", matches_2g.any().item(), matches_2g.nonzero(as_tuple=True)[-1].tolist())

print("\nDONE")
