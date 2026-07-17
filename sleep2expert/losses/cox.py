import torch


class CoxPHLossVectorized(torch.nn.Module):
    def __init__(self, eps=1e-9):
        super().__init__()
        self.eps = eps

    def forward(self, pred, has_label, event_time, is_event):
        if pred.shape != has_label.shape or pred.shape != event_time.shape or pred.shape != is_event.shape:
            raise ValueError("Cox PH tensors must all have shape [B, L].")
        if pred.dim() != 2:
            raise ValueError("Cox PH tensors must be rank-2 [B, L].")

        z = pred.float()
        valid = has_label.float() > 0.5
        t = event_time.float()
        events = (is_event.float() > 0.5) & valid

        neg_big = torch.finfo(z.dtype).min
        sort_key = torch.where(valid, t, neg_big)
        _, idx = sort_key.sort(dim=0, descending=True)

        z_sorted = z.gather(0, idx)
        t_sorted = t.gather(0, idx)
        valid_sorted = valid.gather(0, idx)
        events_sorted = events.gather(0, idx).float()

        z_masked = torch.where(valid_sorted, z_sorted, neg_big)
        log_risk_prefix = torch.logcumsumexp(z_masked, dim=0)

        # Breslow ties use the same denominator for all events at the same time:
        # the risk-set prefix at the end of that sorted time group.
        same_next = valid_sorted[:-1] & valid_sorted[1:] & (t_sorted[:-1] == t_sorted[1:])
        group_end = valid_sorted.clone()
        group_end[:-1] &= ~same_next
        row_idx = torch.arange(z.size(0), device=z.device).view(-1, 1).expand_as(z)
        end_pos = torch.where(group_end, row_idx, torch.full_like(row_idx, z.size(0)))
        shared_end = torch.flip(torch.cummin(torch.flip(end_pos, dims=[0]), dim=0).values, dims=[0])
        log_riskset = log_risk_prefix.gather(0, shared_end.clamp(max=z.size(0) - 1))

        losses = -((z_sorted - log_riskset) * events_sorted)

        event_counts = events_sorted.sum(dim=0)
        per_label = losses.sum(dim=0) / (event_counts + self.eps)
        used = event_counts > 0
        if used.any():
            return per_label[used].mean()
        return pred.sum() * 0.0
