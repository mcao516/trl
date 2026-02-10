import re
import shap
import torch


def first_true_indices(bools, dtype=torch.long):
    """
    Takes an N-dimensional bool tensor and returns an (N-1)-dimensional tensor of integers giving
    the position of the first True in each "row".

    Returns the length of the rows (bools.size(-1)) if no element is True in a given row.
    """
    row_len = bools.size(-1)
    zero_or_index = row_len * (~bools).type(dtype) + torch.arange(row_len, dtype=dtype, device=bools.device)
    return torch.min(zero_or_index, dim=-1).values


def get_shap_rewards(model, query_str, response_str, tokenizer, masker=None):
    def f(x):
        partial_sentences = []
        for _x in x:
            if len(_x) > 0 and _x[0] == " ":
                concatenated = query_str + _x
            else:
                concatenated = query_str + " " + _x
            concatenated += tokenizer.eos_token
            partial_sentences.append(concatenated)

        inputs = tokenizer(partial_sentences, padding="longest", padding_side="right", return_tensors="pt").to("cuda")
        reward_logits = model(
            **inputs,
            return_dict=True,
            output_hidden_states=True,
        )  # reward_logits: [bsz, seq_len, 1]
        sequence_lengths = first_true_indices(inputs["input_ids"] == tokenizer.pad_token_id) - 1
        output = reward_logits[torch.arange(reward_logits.shape[0], device=reward_logits.device), sequence_lengths].squeeze(-1)
        return output.detach().cpu().float().numpy()

    masker = tokenizer if not masker else masker
    explainer = shap.Explainer(f, masker, algorithm="auto")
    shap_values = explainer([response_str])

    return shap_values


def parse_sentence(paragraph, return_offsets_mapping=True):
    """
    Parse a paragraph into spans based on delimiters and return offset mappings.
    
    Args:
        paragraph (str): The input English paragraph.
    
    Returns:
        spans (list): A list of spans split by the delimiters.
        offset_mapping (list): A list of tuples indicating the start and end position of each span.
    """
    # Regex pattern for the delimiters
    pattern = r"[.,;?!]"
    
    spans = []
    offset_mapping = []
    start = 0
    
    for match in re.finditer(pattern, paragraph):
        end = match.end()
        span = paragraph[start:end]
        if span:  # Only add non-empty spans
            spans.append(span)
            offset_mapping.append((start, end))
        start = end
    
    # Add the last span if there's any text left after the final delimiter
    if start < len(paragraph):
        spans.append(paragraph[start:])
        offset_mapping.append((start, len(paragraph)))
    
    if return_offsets_mapping:
        return {'input_ids': spans, 'offset_mapping': offset_mapping}
    else:
        return {'input_ids': spans}