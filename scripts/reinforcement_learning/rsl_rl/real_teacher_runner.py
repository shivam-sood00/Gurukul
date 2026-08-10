from __future__ import annotations

from contact_trail_runner import ContactTrailOnPolicyRunner


class RealTeacherOnPolicyRunner(ContactTrailOnPolicyRunner):
    """On-policy runner that forwards REAL policy attention metrics to the active logger."""

    def _log_contact_trail_metrics(self, iteration: int) -> None:
        policy = getattr(self.alg, "policy", None)
        if policy is None or not hasattr(policy, "pop_logging_stats"):
            return

        metrics = policy.pop_logging_stats()
        writer = getattr(self.logger, "writer", None)
        if writer is None or not metrics:
            return

        for tag, value in metrics.items():
            writer.add_scalar(tag, value, iteration)
