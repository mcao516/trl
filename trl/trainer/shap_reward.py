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
        input_ids = inputs["input_ids"]
        attention_mask = input_ids != tokenizer.pad_token_id
        position_ids = attention_mask.cumsum(1) - attention_mask.long()  # exclusive cumsum
        lm_backbone = getattr(model, model.base_model_prefix)
        input_ids = torch.masked_fill(input_ids, ~attention_mask, 0)
        output = lm_backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=True,
            output_hidden_states=True,
            use_cache=False,  # otherwise mistral-based RM would error out
        )
        reward_logits = model.score(output.hidden_states[-1])

        # reward_logits_ = model(
        #     input_ids=inputs["input_ids"],
        #     attention_mask=inputs["attention_mask"],
        # )
        # output_ = reward_logits_.logits[:, 1] - reward_logits_.logits[:, 0]

        assert len(reward_logits.shape) == 3
        assert reward_logits.shape[-1] == 1 or reward_logits.shape[-1] == 2
        if reward_logits.shape[-1] == 2:
            reward_logits = reward_logits[:, :, 1] - reward_logits[:, :, 0]
            reward_logits = reward_logits.unsqueeze(-1)
        # reward_logits = model(
        #     **inputs,
        #     return_dict=True,
        #     output_hidden_states=True,
        # )  # reward_logits: [bsz, seq_len, 1]
        sequence_lengths = first_true_indices(inputs["input_ids"] == tokenizer.pad_token_id) - 1
        output = reward_logits[torch.arange(reward_logits.shape[0], device=reward_logits.device), sequence_lengths].squeeze(-1)
        return output.detach().cpu().float().numpy()

    masker = tokenizer if not masker else masker
    explainer = shap.Explainer(f, masker, algorithm="auto")
    shap_values = explainer([response_str])

    return shap_values