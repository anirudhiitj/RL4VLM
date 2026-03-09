import torch
import math
import os
def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def llava_generate(value_model, tokenizer, input_ids, image_tensor, args):
    base = value_model.base
    image_tensor = image_tensor.to(base.device, dtype = base.dtype)
    _, _, _, _, inputs_embeds, _ = base.prepare_inputs_labels_for_multimodal(input_ids.to(base.device), None, None, None, None, image_tensor)
    inputs_embeds = inputs_embeds.to(base.device, dtype = base.dtype)
    with torch.inference_mode():
        outputs = base.generate(
        inputs_embeds = inputs_embeds,
        do_sample=True,
        temperature=args.temperature,
        num_beams=args.num_beams,
        max_new_tokens=args.max_new_tokens,
        use_cache=True,
        output_scores=True,
        output_hidden_states=True,
        return_dict_in_generate=True,
        pad_token_id=tokenizer.eos_token_id,)
        output_ids = outputs['sequences']
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    padded_output_ids = torch.zeros(output_ids.size(0), 2*args.max_new_tokens).to(dtype=output_ids.dtype, device = output_ids.device)
    padded_output_ids[:, :output_ids.size(1)] = output_ids
    with torch.no_grad():
        values, sum_log_probs, action_tokens_log_prob = llava_evaluate(value_model, input_ids, padded_output_ids, image_tensor, args.temperature, args.thought_prob_coef)
    return values, padded_output_ids, outputs, sum_log_probs, action_tokens_log_prob

def llava_evaluate(value_model, input_ids, output_ids, image_tensor, temperature, thought_prob_coef):
    if output_ids.size(0) != 1:
        input_ids = input_ids.broadcast_to(output_ids.size(0), input_ids.size(-1))
    base = value_model.base
    image_tensor = image_tensor.to(base.device, dtype=base.dtype)
    output_ids = output_ids.to(base.device)
    input_ids = input_ids.to(base.device)
    _, _, _, _, inputs_embeds, _ = base.prepare_inputs_labels_for_multimodal(torch.cat([input_ids, output_ids], dim = 1), None, None, None, None, image_tensor)

    #calling the model
    inputs_embeds = inputs_embeds.to(base.device, dtype = base.dtype)
    #omit the first output token
    outputs = base(
        inputs_embeds = inputs_embeds,
        output_hidden_states = True,
        )
    scores = outputs.logits

    input_token_len = inputs_embeds.shape[1] - output_ids.shape[1]
    hidden_states = outputs.hidden_states[-1][:, input_token_len-1]
    values = value_model.value_head(hidden_states)
    scores = scores * (1/temperature)
    scores = scores.to(torch.float32)
    log_probs = torch.nn.functional.log_softmax(scores, dim=-1)
    log_probs = log_probs.to(torch.bfloat16)
    output_ids_mask = (output_ids != 0)[:, 1:]
    selected_log_probs = output_ids_mask*torch.take_along_dim(log_probs[:, input_token_len:-1], output_ids[:,1:].unsqueeze(2), dim = 2).squeeze(2)
    unfolded = output_ids.unfold(dimension=-1, size=3, step=1)
    # tokens for '"action":' — ID 345 is " after space/start, ID 28739 is " after newline
    target1 = torch.tensor([345,1774,1264]).to(base.device)
    target2 = torch.tensor([28739,1774,1264]).to(base.device)
    matches = (unfolded == target1).all(dim=-1) | (unfolded == target2).all(dim=-1)
    match_index = matches.nonzero(as_tuple=True)[-1]
    if match_index.shape[0] >= 1:
        match_index = match_index[-1].unsqueeze(0)
    else:
        try:
            match_index = output_ids_mask.nonzero(as_tuple=False)[-4,1]
        except:
            sum_log_prob = torch.tensor([-2]).to(base.device)
            action_tokens_log_prob = torch.tensor([-1]).to(base.device)
            return values, sum_log_prob, action_tokens_log_prob
    ## omitting the second token for calculating log prob, because its logprb is very very small
    thought_log_prob = torch.sum(selected_log_probs[:,1:match_index-1], dim = 1)
    action_tokens_log_prob = torch.sum(selected_log_probs[:,match_index-1:], dim = 1)
    # DEBUG: write critical info to file (first 5 calls only)
    _debug_file = "/tmp/rl_token_debug.txt"
    try:
        _existing = 0
        if os.path.exists(_debug_file):
            with open(_debug_file, "r") as _df:
                _existing = _df.read().count("=== call")
        if _existing < 5:
            nonzero_ids = output_ids[0][output_ids[0] != 0].tolist()
            raw_log_probs = torch.take_along_dim(log_probs[:, input_token_len:-1], output_ids[:,1:].unsqueeze(2), dim = 2).squeeze(2)
            with open(_debug_file, "a") as _df:
                _df.write("=== call {} ===\n".format(_existing + 1))
                _df.write("output_ids non-zero ({}): {}\n".format(len(nonzero_ids), nonzero_ids))
                _df.write("input_token_len: {}\n".format(input_token_len))
                _df.write("inputs_embeds.shape: {}\n".format(list(inputs_embeds.shape)))
                _df.write("log_probs.shape: {}\n".format(list(log_probs.shape)))
                _df.write("output_ids.shape: {}\n".format(list(output_ids.shape)))
                _df.write("selected_log_probs.shape: {}\n".format(list(selected_log_probs.shape)))
                _df.write("match found: {} | match_index: {}\n".format(matches.any().item(), match_index.tolist()))
                mi = match_index.item()
                _df.write("output_ids_mask around match [{}-{}]: {}\n".format(
                    max(0,mi-2), min(output_ids_mask.shape[1], mi+8),
                    output_ids_mask[0, max(0,mi-2):mi+8].tolist()))
                _df.write("raw_log_probs (before mask) [{}-{}]: {}\n".format(
                    max(0,mi-2), min(raw_log_probs.shape[1], mi+8),
                    raw_log_probs[0, max(0,mi-2):mi+8].tolist()))
                _df.write("selected_log_probs [{}-{}]: {}\n".format(
                    max(0,mi-2), min(selected_log_probs.shape[1], mi+8),
                    selected_log_probs[0, max(0,mi-2):mi+8].tolist()))
                _df.write("output_ids around match [{}-{}]: {}\n".format(
                    max(0,mi-1), min(output_ids.shape[1], mi+9),
                    output_ids[0, max(0,mi-1):mi+9].tolist()))
                _df.write("thought_log_prob: {}\n".format(thought_log_prob))
                _df.write("action_tokens_log_prob: {}\n".format(action_tokens_log_prob))
                _df.write("slice used: selected_log_probs[:, {}:]\n".format(mi-1))
                _df.write("sum of selected_log_probs[:, 1:{}]: {}\n".format(mi-1, torch.sum(selected_log_probs[:,1:mi-1], dim=1)))
                _df.write("sum of selected_log_probs[:, {}:]: {}\n".format(mi-1, torch.sum(selected_log_probs[:,mi-1:], dim=1)))
                _df.write("\n")
    except Exception as e:
        with open(_debug_file, "a") as _df:
            _df.write("DEBUG ERROR: {}\n".format(e))
    sum_log_prob = thought_prob_coef*thought_log_prob + action_tokens_log_prob
    return values, sum_log_prob, action_tokens_log_prob
