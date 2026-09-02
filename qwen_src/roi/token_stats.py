"""Query-token hidden-state gather for the SD-RPN twig (per-head Q/K)."""
import torch


def get_singleturn_query_text_hs_mheads(
    qk_states: torch.Tensor,
    labels: torch.Tensor
):
    is_response_token = (labels != -100)  # Shape: (batch_size, sequence_length)
    first_response_token_indices_all_samples = torch.argmax(is_response_token.int(), dim=1)
    first_response_token_indices_all_samples = torch.max(
        torch.tensor(0, device=labels.device),
        first_response_token_indices_all_samples - 1
    )
    query_indices = first_response_token_indices_all_samples

    ##Gather the hidden states for the selected query tokens
    batch, num_heads, sequence_length, hidden_dim = qk_states.shape
    idx_expanded = query_indices.view(batch, 1, 1, 1).expand(batch, num_heads, 1, hidden_dim)
    qk_hidden_states = torch.gather(qk_states, 2, idx_expanded)  # Shape: (batch, num_heads, 1, hidden_dim)

    #qk_hidden_states = qk_states[:,:, first_response_token_indices_all_samples-2:first_response_token_indices_all_samples+1, :]
    return qk_hidden_states, query_indices
