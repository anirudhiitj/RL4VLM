"""
Debug script: Why is action_tokens_log_prob always 0?
Traces through the exact same code path as llava_evaluate.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'VLM_PPO'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'gym-cards'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'LLaVA'))

# Must import xformers patch BEFORE llava to avoid flash_attn ABI issues
import xformers
from patch import replace_llama_attn_with_xformers_attn
replace_llama_attn_with_xformers_attn()
print("using xformers")

import torch
import numpy as np
import gymnasium as gym
import gym_cards

from transformers import AutoTokenizer
from llava.model.language_model.llava_mistral import LlavaMistralForCausalLM
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token

from a2c_ppo_acktr.llava_interface.utils import init_pretrained_model
from a2c_ppo_acktr.rl_utils import get_prompt

TEMPERATURE = 0.2
MAX_NEW_TOKENS = 256
THOUGHT_PROB_COEF = 0.5

def main():
    model_path = "/mnt/raid/rl_gaming/RL4VLM/checkpoints/sft_numberline"
    print(f"Loading model from {model_path} ...")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    base = LlavaMistralForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="flash_attention_2"
    )
    base, tokenizer = init_pretrained_model(base, tokenizer)
    base.eval()

    image_processor = base.get_vision_tower().image_processor
    device = base.device

    # Build prompt (same as RL training)
    qs = get_prompt("gym_cards/NumberLine-v0", action_only=False)
    qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
    conv = conv_templates["llava_v1"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    INPUT_IDS = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)
    INPUT_IDS[INPUT_IDS == 0] = 259

    # Create environment
    env = gym.make("gym_cards/NumberLine-v0")
    obs, info = env.reset(seed=42)

    # Process image
    img_tensor = image_processor.preprocess(obs, return_tensors='pt')['pixel_values'].to(device=device, dtype=base.dtype)

    # =============================================
    # STEP 1: Generate (same as llava_generate)
    # =============================================
    input_ids = INPUT_IDS.to(device)
    _, _, _, _, inputs_embeds, _ = base.prepare_inputs_labels_for_multimodal(
        input_ids, None, None, None, None, img_tensor
    )
    inputs_embeds = inputs_embeds.to(device=device, dtype=base.dtype)

    with torch.inference_mode():
        outputs = base.generate(
            inputs_embeds=inputs_embeds,
            do_sample=True,
            temperature=TEMPERATURE,
            num_beams=1,
            max_new_tokens=MAX_NEW_TOKENS,
            use_cache=True,
            output_scores=True,
            output_hidden_states=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        output_ids = outputs['sequences']

    text_output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    print(f"\n{'='*60}")
    print(f"Generated text: {text_output}")
    print(f"{'='*60}")

    print(f"\noutput_ids shape: {output_ids.shape}")
    print(f"output_ids first 5: {output_ids[0, :5].tolist()}")
    print(f"output_ids last 5 non-zero: ", end="")
    nonzero_mask = output_ids[0] != 0
    nonzero_ids = output_ids[0][nonzero_mask]
    print(f"{nonzero_ids[-5:].tolist()}")
    print(f"Total non-zero tokens in output_ids: {nonzero_mask.sum().item()}")

    # Pad output_ids (same as llava_generate)
    padded_output_ids = torch.zeros(output_ids.size(0), 2*MAX_NEW_TOKENS).to(dtype=output_ids.dtype, device=output_ids.device)
    padded_output_ids[:, :output_ids.size(1)] = output_ids

    print(f"\npadded_output_ids shape: {padded_output_ids.shape}")
    print(f"Non-zero count in padded: {(padded_output_ids[0] != 0).sum().item()}")

    # =============================================
    # STEP 2: Evaluate (same as llava_evaluate)
    # =============================================
    output_ids_eval = padded_output_ids  # rename for clarity

    # Concat input_ids + output_ids
    cat_ids = torch.cat([input_ids, output_ids_eval], dim=1)
    print(f"\ncat_ids shape: {cat_ids.shape}")

    # Get multimodal embeddings
    _, _, _, _, inputs_embeds_eval, _ = base.prepare_inputs_labels_for_multimodal(
        cat_ids, None, None, None, None, img_tensor
    )
    inputs_embeds_eval = inputs_embeds_eval.to(device=device, dtype=base.dtype)

    print(f"inputs_embeds_eval shape: {inputs_embeds_eval.shape}")

    # Forward pass
    with torch.no_grad():
        eval_outputs = base(inputs_embeds=inputs_embeds_eval, output_hidden_states=True)
    scores = eval_outputs.logits

    input_token_len = inputs_embeds_eval.shape[1] - output_ids_eval.shape[1]
    print(f"\ninput_token_len (expanded prompt): {input_token_len}")
    print(f"output_ids_eval shape[1]: {output_ids_eval.shape[1]}")
    print(f"scores shape: {scores.shape}")

    # Compute log probs
    scores_scaled = scores * (1/TEMPERATURE)
    scores_scaled = scores_scaled.to(torch.float32)
    log_probs = torch.nn.functional.log_softmax(scores_scaled, dim=-1)
    log_probs = log_probs.to(torch.bfloat16)

    output_ids_mask = (output_ids_eval != 0)[:, 1:]
    print(f"\noutput_ids_mask shape: {output_ids_mask.shape}")
    print(f"output_ids_mask True count: {output_ids_mask.sum().item()}")

    log_probs_slice = log_probs[:, input_token_len:-1]
    print(f"log_probs_slice shape: {log_probs_slice.shape}")

    output_ids_shifted = output_ids_eval[:, 1:]
    print(f"output_ids_shifted shape: {output_ids_shifted.shape}")

    selected_log_probs = output_ids_mask * torch.take_along_dim(
        log_probs_slice, output_ids_shifted.unsqueeze(2), dim=2
    ).squeeze(2)

    print(f"selected_log_probs shape: {selected_log_probs.shape}")
    print(f"selected_log_probs non-zero count: {(selected_log_probs[0] != 0).sum().item()}")
    print(f"selected_log_probs first 10: {selected_log_probs[0, :10].tolist()}")

    # =============================================
    # STEP 3: Find "action": token match
    # =============================================
    target = torch.tensor([345, 1774, 1264]).to(device)
    unfolded = output_ids_eval.unfold(dimension=-1, size=3, step=1)
    matches = (unfolded == target).all(dim=-1)

    print(f"\n{'='*60}")
    print(f"Searching for [345, 1774, 1264] in padded_output_ids...")
    print(f"matches shape: {matches.shape}")
    print(f"Number of matches: {matches.sum().item()}")

    match_nonzero = matches.nonzero(as_tuple=True)
    print(f"match_nonzero: batch={match_nonzero[0].tolist()}, pos={match_nonzero[-1].tolist()}")

    match_index = match_nonzero[-1]
    if match_index.shape[0] >= 1:
        match_index = match_index[-1].unsqueeze(0)
        print(f"match_index (last match): {match_index.item()}")
    else:
        print("NO MATCH FOUND!")
        return

    # Show tokens around the match
    M = match_index.item()
    print(f"\nTokens around match position {M} in output_ids_eval:")
    for i in range(max(0, M-3), min(output_ids_eval.shape[1], M+8)):
        tid = output_ids_eval[0, i].item()
        decoded = tokenizer.decode([tid]) if tid != 0 else '<PAD>'
        print(f"  pos {i}: id={tid} -> {decoded!r}")

    # =============================================
    # STEP 4: Compute thought & action log probs
    # =============================================
    print(f"\n{'='*60}")
    print(f"Computing log prob split at match_index={M}")
    print(f"thought_log_prob = sum(selected_log_probs[:, 1:{M-1}])")
    print(f"action_tokens_log_prob = sum(selected_log_probs[:, {M-1}:])")

    thought_region = selected_log_probs[:, 1:M-1]
    action_region = selected_log_probs[:, M-1:]

    print(f"\nthought_region shape: {thought_region.shape}")
    print(f"thought_region non-zero: {(thought_region[0] != 0).sum().item()}")
    print(f"thought_log_prob = {thought_region.sum(dim=1).item():.6f}")

    print(f"\naction_region shape: {action_region.shape}")
    print(f"action_region non-zero: {(action_region[0] != 0).sum().item()}")
    print(f"action_region first 15 values: {action_region[0, :15].tolist()}")
    print(f"action_tokens_log_prob = {action_region.sum(dim=1).item():.6f}")

    thought_log_prob = thought_region.sum(dim=1)
    action_tokens_log_prob = action_region.sum(dim=1)
    sum_log_prob = THOUGHT_PROB_COEF * thought_log_prob + action_tokens_log_prob

    print(f"\n{'='*60}")
    print(f"FINAL RESULTS:")
    print(f"  thought_log_prob:        {thought_log_prob.item():.6f}")
    print(f"  action_tokens_log_prob:  {action_tokens_log_prob.item():.6f}")
    print(f"  sum_log_prob:            {sum_log_prob.item():.6f}")
    print(f"  (lambda={THOUGHT_PROB_COEF})")
    print(f"{'='*60}")

    env.close()

if __name__ == "__main__":
    main()
