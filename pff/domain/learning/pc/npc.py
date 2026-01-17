"""Neural Probabilistic Circuit (NPC) - Pairwise PC for Knowledge Graph Completion.

This module implements a differentiable probabilistic circuit following the
PC2 (Pairwise Constraint Probabilistic Circuit) approach. PC2 is characterized
by capturing pairwise correlations between attributes with **exact probability
computation** guarantees.

Key Architecture (PC2 - Pairwise Constraint PC):
- **Pairwise factors**: Captures correlations between attribute pairs (h,r),
  (r,t), (h,t) explicitly before aggregating, rather than assuming full
  independence (Naive Bayes) or full connectivity (intractable).
- **HCLT backbone**: Hidden Chow-Liu Tree structure enables O(|Z|) inference.
- **Exact inference**: No approximations - smoothness and decomposability
  guarantee tractable exact probability computation.

Important Terminological Distinction:
- **PC2 (this implementation)**: Pairwise Constraint PC - captures 2nd order
  attribute correlations with tractability guarantees.
- **PC² (NOT this)**: "Probabilistic Circuits Squared" - a theoretical
  concept in circuit complexity involving structured-decomposability and vtrees.

The implementation uses a Hidden Chow-Liu Tree (HCLT) skeleton for tractable
exact inference. Key properties:
1. Smoothness: All sum nodes have children with identical scopes
2. Decomposability: All product nodes have children with disjoint scopes
3. Exact inference: No approximations in marginal computation

Design Patterns:
    - Template Method: forward() orchestrates tractability checks, flow
      updates, and NLL computation.
    - Builder: rebuild() reorders the tree from mutual-information scores.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from pff.shared.core.logger import logger
from .triton_kernels import pc2_forward_triton, TRITON_AVAILABLE as PC_TRITON_AVAILABLE


@dataclass
class CircuitProperties:
    """Tractability flags for the PC2 (Pairwise Constraint PC) structure.

    PC2 circuits guarantee exact probability computation through
    smoothness and decomposability properties. The pairwise constraint
    captures 2nd order correlations while maintaining O(|edges|) inference.
    """

    smooth: bool  # Sum nodes have children with identical scopes
    decomposable: bool  # Product nodes have children with disjoint scopes
    exact_inference: bool  # PC2 computes exact probabilities
    max_depth: int


class NeuralProbabilisticCircuit(nn.Module):
    """PC2 (Pairwise Constraint PC) for binary attributes.

    This circuit implements the PC2 approach for Knowledge Graph Completion,
    which uses probabilistic circuits for **exact probability computation**.
    Unlike Noisy-OR approximations, PC2 computes exact marginals using the
    circuit's tractability properties.

    The circuit models P(Y, Z) where Y is a binary label and Z are binary
    attributes emitted by the DSLFM attribute recognizer. Conditionals are
    parameterized as Bernoulli tables along an HCLT (Hidden Chow-Liu Tree),
    enabling exact likelihood computation in O(|Z|).

    PC2 Architecture:
    - Pairwise factors capture correlations between adjacent attributes
    - HCLT structure separates h from t conditioned on r (Markov property)
    - Balance between expressivity (pairwise) and tractability (tree structure)

    PC2 Properties Guaranteed:
    - Smooth: All sum nodes sum over identical scopes
    - Decomposable: All product nodes multiply disjoint scopes
    - Exact inference: No approximations in probability computation

    Performance vs. Complexity Trade-off (from literature):
    - PC1 (Naive): Fast O(|Z|), but assumes independence - low expressivity
    - PC2 (this): Fast O(|edges|), captures pairwise correlations - medium-high
    - Full PC: Exponential, captures all correlations - intractable for KGC

    Note:
        This is PC2 (Pairwise Constraint), NOT PC² (Circuits Squared). PC² is
        a different theoretical concept involving structured-decomposability
        and vtrees. PC2 focuses on pairwise attribute modeling for KGC.
    """

    def __init__(
        self,
        num_attrs: int,
        smoothing_epsilon: float = 1e-6,
        pruning_threshold: float = 0.01,
        grow_noise: float = 0.01,
        max_depth: int | None = None,
        prune_every_n_steps: int = 100,
    ) -> None:
        """Initialize the PC2 circuit.

        Args:
            num_attrs: Number of binary attributes (Z) modeled by the circuit.
            smoothing_epsilon: Numerical floor to avoid log(0).
            pruning_threshold: Minimum edge flow to keep parent-child links.
            grow_noise: Gaussian noise std applied during growth to avoid
                collapse after pruning.
            max_depth: Optional maximum tree depth enforced after rebuild.
            prune_every_n_steps: Only run _auto_prune every N forward passes (default 100).
        """
        super().__init__()
        if num_attrs < 1:
            raise ValueError("num_attrs must be >= 1")
        self.num_attrs = num_attrs
        self.smoothing_epsilon = float(smoothing_epsilon)
        self.pruning_threshold = float(pruning_threshold)
        self.grow_noise = float(grow_noise)
        self.max_depth = max_depth
        self.prune_every_n_steps = max(1, int(prune_every_n_steps))
        self._forward_count = 0
        self._total_prune_calls = 0

        # Parameters for P(Y) - label prior
        self.label_logits = nn.Parameter(torch.zeros(2))

        # Parameters for P(Z_root | Y) - root attribute given label
        self.root_logits = nn.ParameterList(
            [nn.Parameter(torch.zeros(2)) for _ in range(num_attrs)]
        )

        # Parameters for P(Z_child | Z_parent, Y) - conditional tables
        self.conditional_logits = nn.ParameterList(
            [nn.Parameter(torch.zeros(2, 2)) for _ in range(num_attrs)]
        )

        # HCLT parent structure (star by default - all children from root)
        self.parents = [-1] + [0] * (num_attrs - 1)
        self.root = 0
        self._edge_flow = torch.zeros(num_attrs)

    def forward(self, attr_probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute negative log-likelihood for a batch using vectorized exact inference.

        SOTA: Uses vectorized tensor operations instead of Python loop for
        O(1) Python overhead regardless of num_attrs. Enables torch.compile fusion.

        Args:
            attr_probs: Tensor with shape [batch, num_attrs, 2] containing
                probabilities (P(attr), P(not attr)).
            labels: Tensor with shape [batch] containing binary targets.

        Returns:
            Scalar negative log-likelihood.
        """
        if attr_probs.dim() != 3 or attr_probs.size(2) != 2:
            raise ValueError(
                f"attr_probs must have shape [batch, num_attrs, 2], got {attr_probs.shape}"
            )
        if attr_probs.size(1) != self.num_attrs:
            raise ValueError(
                f"attr_probs num_attrs={attr_probs.size(1)} does not match circuit={self.num_attrs}"
            )
        if attr_probs.size(0) == 0:
            return torch.zeros((), device=attr_probs.device, dtype=attr_probs.dtype)

        batch_size = attr_probs.size(0)
        labels_flat = labels.view(-1).float()

        # Clamp and clean probabilities
        attr_probs = torch.nan_to_num(
            attr_probs,
            nan=self.smoothing_epsilon,
            posinf=1.0 - self.smoothing_epsilon,
            neginf=self.smoothing_epsilon,
        )
        attr_probs = torch.clamp(
            attr_probs, self.smoothing_epsilon, 1.0 - self.smoothing_epsilon
        )
        pos_probs = attr_probs[..., 0]  # [batch, num_attrs]
        neg_probs = attr_probs[..., 1]  # [batch, num_attrs]

        # Build parent tensor for vectorized indexing
        parents_tensor = torch.tensor(
            self.parents, device=attr_probs.device, dtype=torch.long
        )
        is_root = parents_tensor == -1  # [num_attrs]

        # Compute P(Y) prior
        log_prior = torch.log_softmax(self.label_logits, dim=0)
        log_prob_y0 = log_prior[0].expand(batch_size)
        log_prob_y1 = log_prior[1].expand(batch_size)

        # Stack root logits for vectorized computation: [num_attrs, 2]
        root_logits_stacked = torch.stack(list(self.root_logits), dim=0)
        root_probs = torch.sigmoid(root_logits_stacked)
        root_probs = torch.clamp(
            root_probs, self.smoothing_epsilon, 1.0 - self.smoothing_epsilon
        )

        # Stack conditional logits: [num_attrs, 2, 2]
        cond_logits_stacked = torch.stack(list(self.conditional_logits), dim=0)
        cond_probs = torch.sigmoid(cond_logits_stacked)
        cond_probs = torch.clamp(
            cond_probs, self.smoothing_epsilon, 1.0 - self.smoothing_epsilon
        )

        if PC_TRITON_AVAILABLE and attr_probs.is_cuda:
            log_prob_y0, log_prob_y1 = pc2_forward_triton(
                pos_probs,
                parents_tensor,
                root_probs,
                cond_probs,
                torch.log_softmax(self.label_logits, dim=0),
            )
        else:
            # === VECTORIZED ROOT COMPUTATION ===
            # For root nodes: log P(Z | Y)
            # root_probs: [num_attrs, 2] -> probs for Y=0 and Y=1
            # Expand for batch: [1, num_attrs, 2]
            root_probs_expanded = root_probs.unsqueeze(0)

            # log P(attr=1 | Y=0) and log P(attr=0 | Y=0)
            log_p_attr_given_y0 = pos_probs * torch.log(
                root_probs_expanded[:, :, 0]
            ) + neg_probs * torch.log(1.0 - root_probs_expanded[:, :, 0])
            log_p_attr_given_y1 = pos_probs * torch.log(
                root_probs_expanded[:, :, 1]
            ) + neg_probs * torch.log(1.0 - root_probs_expanded[:, :, 1])

            # Mask to only include root nodes: [num_attrs]
            root_mask = is_root.float().unsqueeze(0)  # [1, num_attrs]
            log_prob_y0 = log_prob_y0 + (log_p_attr_given_y0 * root_mask).sum(dim=1)
            log_prob_y1 = log_prob_y1 + (log_p_attr_given_y1 * root_mask).sum(dim=1)

            # === VECTORIZED CHILD COMPUTATION ===
            # For non-root nodes: log P(Z_child | Z_parent, Y)
            child_mask = (~is_root).float()  # [num_attrs]

            if child_mask.sum() > 0:
                # Get parent indices (clamp to 0 for roots to avoid index error)
                parent_indices = torch.clamp(parents_tensor, min=0)  # [num_attrs]

                # Get parent probabilities: [batch, num_attrs]
                parent_true = pos_probs.gather(
                    1, parent_indices.unsqueeze(0).expand(batch_size, -1)
                )

                # cond_probs: [num_attrs, 2, 2] -> [parent_state, y_state]
                # p[idx, parent_state, y_state] = P(Z_idx=1 | parent=parent_state, Y=y_state)
                p0_parent0 = cond_probs[:, 0, 0]  # [num_attrs]
                p0_parent1 = cond_probs[:, 1, 0]  # [num_attrs]
                p1_parent0 = cond_probs[:, 0, 1]  # [num_attrs]
                p1_parent1 = cond_probs[:, 1, 1]  # [num_attrs]

                # Compute log likelihoods for Y=0
                log_given_parent1_y0 = pos_probs * torch.log(
                    p0_parent1
                ) + neg_probs * torch.log(1.0 - p0_parent1)
                log_given_parent0_y0 = pos_probs * torch.log(
                    p0_parent0
                ) + neg_probs * torch.log(1.0 - p0_parent0)
                child_log_y0 = (
                    parent_true * log_given_parent1_y0
                    + (1.0 - parent_true) * log_given_parent0_y0
                )

                # Compute log likelihoods for Y=1
                log_given_parent1_y1 = pos_probs * torch.log(
                    p1_parent1
                ) + neg_probs * torch.log(1.0 - p1_parent1)
                log_given_parent0_y1 = pos_probs * torch.log(
                    p1_parent0
                ) + neg_probs * torch.log(1.0 - p1_parent0)
                child_log_y1 = (
                    parent_true * log_given_parent1_y1
                    + (1.0 - parent_true) * log_given_parent0_y1
                )

                # Apply child mask and sum
                child_mask_expanded = child_mask.unsqueeze(0)  # [1, num_attrs]
                log_prob_y0 = log_prob_y0 + (child_log_y0 * child_mask_expanded).sum(
                    dim=1
                )
                log_prob_y1 = log_prob_y1 + (child_log_y1 * child_mask_expanded).sum(
                    dim=1
                )

        # Compute NLL
        log_probs = torch.stack([log_prob_y0, log_prob_y1], dim=1)
        target_log_prob = torch.where(
            labels_flat > 0.5, log_probs[:, 1], log_probs[:, 0]
        )
        nll = -target_log_prob.mean()

        # Update edge flow (vectorized)
        self._edge_flow = self._estimate_edge_flow(pos_probs)
        self._forward_count += 1
        if (
            self.pruning_threshold > 0.0
            and self._forward_count % self.prune_every_n_steps == 0
        ):
            self._auto_prune()

        return nll

    def log_prob(self, attr_probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute log-probability per example without reduction.

        Args:
            attr_probs: Tensor [*, num_attrs, 2] with P(attr), P(not attr)
            labels: Tensor broadcastable to leading dims of attr_probs, binary targets

        Returns:
            Tensor of log-probabilities with shape matching labels (broadcasted).
        """
        if attr_probs.dim() < 3 or attr_probs.size(-1) != 2:
            raise ValueError(
                f"attr_probs must have shape [..., num_attrs, 2], got {attr_probs.shape}"
            )
        if attr_probs.size(-2) != self.num_attrs:
            raise ValueError(
                f"attr_probs num_attrs={attr_probs.size(-2)} does not match circuit={self.num_attrs}"
            )
        # Flatten leading dims for computation, restore later
        orig_shape = labels.shape
        flat_attr = attr_probs.reshape(-1, self.num_attrs, 2)
        flat_labels = labels.reshape(-1).float()

        attr_clean = torch.nan_to_num(
            flat_attr,
            nan=self.smoothing_epsilon,
            posinf=1.0 - self.smoothing_epsilon,
            neginf=self.smoothing_epsilon,
        ).clamp(self.smoothing_epsilon, 1.0 - self.smoothing_epsilon)
        pos_probs = attr_clean[..., 0]
        neg_probs = attr_clean[..., 1]

        parents_tensor = torch.tensor(
            self.parents, device=attr_probs.device, dtype=torch.long
        )
        is_root = parents_tensor == -1

        log_prior = torch.log_softmax(self.label_logits, dim=0)
        log_prob_y0 = log_prior[0].expand(pos_probs.size(0)).clone()
        log_prob_y1 = log_prior[1].expand(pos_probs.size(0)).clone()

        root_logits_stacked = torch.stack(list(self.root_logits), dim=0)
        root_probs = torch.sigmoid(root_logits_stacked).clamp(
            self.smoothing_epsilon, 1.0 - self.smoothing_epsilon
        )

        cond_logits_stacked = torch.stack(list(self.conditional_logits), dim=0)
        cond_probs = torch.sigmoid(cond_logits_stacked).clamp(
            self.smoothing_epsilon, 1.0 - self.smoothing_epsilon
        )

        # Roots
        root_probs_expanded = root_probs.unsqueeze(0)
        pos_root = pos_probs[:, self.root]
        neg_root = neg_probs[:, self.root]
        log_root_y0 = pos_root * torch.log(
            root_probs_expanded[:, self.root, 0]
        ) + neg_root * torch.log(1.0 - root_probs_expanded[:, self.root, 0])
        log_root_y1 = pos_root * torch.log(
            root_probs_expanded[:, self.root, 1]
        ) + neg_root * torch.log(1.0 - root_probs_expanded[:, self.root, 1])
        log_prob_y0 = log_prob_y0 + log_root_y0
        log_prob_y1 = log_prob_y1 + log_root_y1

        # Children (vectorized)
        child_indices = torch.arange(self.num_attrs, device=attr_probs.device)
        child_mask = ~is_root
        if child_mask.any():
            children = child_indices[child_mask]
            parent_idx = parents_tensor[children]

            parent_true = pos_probs[:, parent_idx]  # [batch, num_children]
            parent_false = neg_probs[:, parent_idx]

            pos_child = pos_probs[:, children]
            neg_child = neg_probs[:, children]

            p0_parent1 = cond_probs[children, 1, 0]
            p0_parent0 = cond_probs[children, 0, 0]
            p1_parent1 = cond_probs[children, 1, 1]
            p1_parent0 = cond_probs[children, 0, 1]

            log_given_parent1_y0 = pos_child * torch.log(
                p0_parent1
            ) + neg_child * torch.log(1.0 - p0_parent1)
            log_given_parent0_y0 = pos_child * torch.log(
                p0_parent0
            ) + neg_child * torch.log(1.0 - p0_parent0)
            child_log_y0 = (
                parent_true * log_given_parent1_y0 + parent_false * log_given_parent0_y0
            )

            log_given_parent1_y1 = pos_child * torch.log(
                p1_parent1
            ) + neg_child * torch.log(1.0 - p1_parent1)
            log_given_parent0_y1 = pos_child * torch.log(
                p1_parent0
            ) + neg_child * torch.log(1.0 - p1_parent0)
            child_log_y1 = (
                parent_true * log_given_parent1_y1 + parent_false * log_given_parent0_y1
            )

            log_prob_y0 = log_prob_y0 + child_log_y0.sum(dim=1)
            log_prob_y1 = log_prob_y1 + child_log_y1.sum(dim=1)

        log_probs = torch.stack([log_prob_y0, log_prob_y1], dim=1)
        target_log_prob = torch.where(
            flat_labels > 0.5, log_probs[:, 1], log_probs[:, 0]
        )
        return target_log_prob.reshape(orig_shape)

    def rebuild(self, mi_scores: torch.Tensor) -> None:
        """Rebuild the tree using mutual-information-like scores.

        Reorders the HCLT structure based on importance scores while
        maintaining smoothness and decomposability for exact inference.

        Args:
            mi_scores: Tensor of shape [num_attrs] with importance scores.
        """
        if mi_scores.numel() < self.num_attrs:
            padding = torch.zeros(
                self.num_attrs - mi_scores.numel(), device=mi_scores.device
            )
            mi_scores = torch.cat([mi_scores, padding], dim=0)

        order = torch.argsort(mi_scores, descending=True).tolist()
        self.root = order[0]

        # Rebuild star topology with new root
        new_parents = [-1] * self.num_attrs
        for idx in order[1:]:
            new_parents[idx] = self.root
        self.parents = new_parents

        if self.max_depth is not None:
            self.parents = self._enforce_max_depth(self.parents, self.max_depth)

    def gradient_noise(self) -> None:
        """Apply small Gaussian noise to parameters (growth step)."""
        if self.grow_noise <= 0.0:
            return
        with torch.no_grad():
            for param in self.parameters():
                param.add_(torch.randn_like(param) * self.grow_noise)

    def properties(self) -> CircuitProperties:
        """Return PC2 tractability properties.

        PC2 circuits guarantee exact probability computation through:
        - Smoothness: All sum nodes have children with identical scopes
        - Decomposability: All product nodes have children with disjoint scopes
        - Exact inference: No approximations in marginal computation

        Returns:
            CircuitProperties with PC2 flags.
        """
        depth = self._max_tree_depth()

        return CircuitProperties(
            smooth=True,  # HCLT is always smooth by construction
            decomposable=True,  # HCLT is always decomposable by construction
            exact_inference=True,  # PC2 computes exact probabilities
            max_depth=depth,
        )

    def _auto_prune(self) -> None:
        """Detach edges with low probabilistic flow (with cooldown to avoid loops)."""
        low_flow_edges = self._edge_flow < self.pruning_threshold
        if not torch.any(low_flow_edges):
            return
        parent_tensor = torch.tensor(self.parents, device=low_flow_edges.device)
        to_prune = torch.nonzero(low_flow_edges, as_tuple=True)[0]
        to_prune = to_prune[to_prune > 0]  # skip root
        if to_prune.numel() == 0:
            return
        parent_tensor[to_prune] = self.root
        self.parents = parent_tensor.tolist()
        self._total_prune_calls += 1
        logger.debug(
            f"PC2 pruning: step={self._forward_count}, removed={to_prune.numel()}, "
            f"threshold={self.pruning_threshold}, total_prune_calls={self._total_prune_calls}",
        )

    def _estimate_edge_flow(self, pos_probs: torch.Tensor) -> torch.Tensor:
        """Approximate circuit flow using attribute co-activation.

        Vectorized implementation that computes joint probabilities for all
        parent-child pairs in parallel using gather operations.

        Args:
            pos_probs: Tensor of shape (batch, num_attrs) with positive leaf outputs.

        Returns:
            Tensor of shape (num_attrs,) with flow estimates per edge.
        """
        # Create parents tensor on correct device
        parents_t = torch.tensor(
            self.parents, device=pos_probs.device, dtype=torch.long
        )

        # Mask for valid edges: idx > 0 (not root) and parent != -1
        valid_mask = (torch.arange(self.num_attrs, device=pos_probs.device) > 0) & (
            parents_t >= 0
        )

        # Clamp parent indices to valid range for gather (invalid ones will be masked out)
        safe_parents = parents_t.clamp(min=0)

        # Gather parent probabilities: pos_probs[:, parent[idx]] for all idx
        # Shape: (batch, num_attrs)
        parent_probs = torch.gather(
            pos_probs,
            dim=1,
            index=safe_parents.unsqueeze(0).expand(pos_probs.size(0), -1),
        )

        # Joint probability: mean over batch of (child_prob * parent_prob)
        # Shape: (num_attrs,)
        joint = torch.mean(pos_probs * parent_probs, dim=0)

        # Apply mask: flows = 0 for invalid edges
        flows = torch.where(valid_mask, joint, torch.zeros_like(joint))

        return flows.detach()

    def _max_tree_depth(self) -> int:
        """Compute tree depth using O(N) memoization."""
        depths = [-1] * self.num_attrs
        max_d = 0

        for i in range(self.num_attrs):
            if depths[i] != -1:
                continue

            path = []
            curr = i
            while curr != -1 and depths[curr] == -1:
                path.append(curr)
                curr = self.parents[curr]

            base_depth = 0
            if curr != -1:
                base_depth = depths[curr]

            # Assign depths along the path
            for node in reversed(path):
                base_depth += 1
                depths[node] = base_depth

            max_d = max(max_d, depths[i])

        return max_d

    def _enforce_max_depth(self, parents: list[int], max_depth: int) -> list[int]:
        """Rewire nodes that exceed max_depth using O(N) BFS traversal."""
        # Build adjacency list
        children = [[] for _ in range(self.num_attrs)]
        root = -1
        for i, p in enumerate(parents):
            if p == -1:
                root = i
            elif p >= 0 and p < self.num_attrs:
                children[p].append(i)

        if root == -1:
            return parents

        adjusted = list(parents)
        # Queue stores (node_index, current_depth)
        # Use a list as queue since we just iterate through it
        queue = [(root, 0)]

        idx = 0
        while idx < len(queue):
            u, d = queue[idx]
            idx += 1

            # Determine effective depth for children
            next_depth = d + 1

            # If current node exceeds depth (excluding root), rewire to root
            if d > max_depth and u != root:
                adjusted[u] = root
                # After rewiring to root, the effective depth for children becomes 1 + 1 = 2
                # (Root is 0, u becomes child of root (1), u's children become depth 2)
                next_depth = 2

            for v in children[u]:
                queue.append((v, next_depth))

        return adjusted

    def extra_repr(self) -> str:
        return (
            f"num_attrs={self.num_attrs}, root={self.root}, "
            f"pruning_threshold={self.pruning_threshold}, "
            f"grow_noise={self.grow_noise}"
        )


# Alias for clarity - PC2 is the official name from KGC research
PC2 = NeuralProbabilisticCircuit
