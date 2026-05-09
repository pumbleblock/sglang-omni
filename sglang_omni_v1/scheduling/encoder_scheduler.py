# SPDX-License-Identifier: Apache-2.0
"""TP-aware encoder scheduler.

Owns the leader/follower coordination for encoder stages backed by the
SGLang-native encoder worker. Same ``inbox`` / ``outbox`` /
``start`` / ``stop`` / ``abort`` shape as
:class:`sglang_omni_v1.scheduling.simple_scheduler.SimpleScheduler` and
:class:`sglang_omni_v1.scheduling.omni_scheduler.OmniScheduler`, so
:class:`Stage` doesn't need a scheduler-type branch.

See sglang-project/sglang-omni#375 design ("EncoderScheduler",
"Inputs Across TP Ranks") for the full contract — this implementation
follows the design verbatim with one practical difference: the
single-rank lane (``tp_size == 1``) skips the broadcast and every TP
collective; the multi-rank lane is the full design.
"""

from __future__ import annotations

import collections
import logging
import os
import queue as _queue_mod
import time
from dataclasses import dataclass
from typing import Any, Callable

import torch

from sglang_omni_v1.pipeline.relay_io import extract_tensors, restore_tensors
from sglang_omni_v1.scheduling.messages import IncomingMessage, OutgoingMessage

logger = logging.getLogger(__name__)


# Picklable tagged sentinel that survives the ``broadcast_pyobj`` pickle
# round-trip — singleton identity (``object()``) does not. Followers
# detect a pre-broadcast leader failure via the ``"kind"`` string.
_RECV_ERROR_KIND = "encoder_recv_error"


@dataclass(slots=True)
class _TensorSpec:
    """Describes a tensor lifted out of ``IncomingMessage.data.data``.

    Carries a real :class:`torch.dtype` (not a stringified one) so the
    follower can call ``torch.empty(shape, dtype=...)`` without an extra
    parser. The placeholder dict ``extract_tensors`` produces stringifies
    dtype on the fly, which is fine for the relay round-trip but breaks
    a NCCL receive — we keep the typed spec parallel to the metadata
    pickle for that reason.
    """

    path: str
    shape: tuple[int, ...]
    dtype: torch.dtype


class BatchCollectError(RuntimeError):
    """Raised when ``request_cost_fn`` faults after draining requests.

    Carries the partial drained batch so :func:`_recv_messages` can
    surface the failure as a per-request recv error rather than silently
    dropping the message that caused the cost-fn fault.
    """

    def __init__(self, messages: list[IncomingMessage], error: BaseException):
        super().__init__(str(error))
        self.messages = messages
        self.error = error


class EncoderScheduler:
    """TP-aware non-AR scheduler for encoder stages.

    Args:
        worker: An :class:`SGLangEncoderWorker` (or any object exposing
            ``tp_rank``, ``tp_size``, ``is_entry_rank``, ``device``,
            ``tp_group``, ``encode_batch``).
        adapter: Implements ``build_batch``, ``run_feature``,
            ``slice_results`` (see Cheng's design,
            "EncoderAdapter and BatchPlan").
        max_batch_size: Cap on requests per forward.
        max_batch_wait_ms: Time the entry rank waits for additional
            requests after the first one arrives.
        request_cost_fn: Optional ``payload -> int`` cost estimator
            applied during admission control.
        max_batch_cost: Optional summed-cost cap.
    """

    def __init__(
        self,
        worker: Any,
        adapter: Any,
        *,
        max_batch_size: int = 32,
        max_batch_wait_ms: int = 50,
        request_cost_fn: Callable[[Any], int] | None = None,
        max_batch_cost: int | None = None,
    ) -> None:
        self.worker = worker
        self.adapter = adapter
        self.inbox: _queue_mod.Queue[IncomingMessage] = _queue_mod.Queue()
        self.outbox: _queue_mod.Queue[OutgoingMessage] = _queue_mod.Queue()
        self._max_batch_size = max(int(max_batch_size), 1)
        self._max_batch_wait_s = max(float(max_batch_wait_ms), 0.0) / 1000.0
        self._request_cost_fn = request_cost_fn
        self._max_batch_cost = (
            max(int(max_batch_cost), 0) if max_batch_cost is not None else None
        )
        self._pending_messages: collections.deque[IncomingMessage] = collections.deque()
        self._aborted_ids: set[str] = set()
        self._running = False

    # ------------------------------------------------------------------
    # Public lifecycle (matches SimpleScheduler / OmniScheduler shape)
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Run the processing loop (blocks the thread)."""
        self._running = True
        try:
            while self._running:
                self._loop_once()
        finally:
            self._running = False

    def stop(self) -> None:
        self._running = False

    def abort(self, request_id: str) -> None:
        """Mark a request as aborted; subsequent results are dropped.

        Encoder stages run a single forward per request, so aborts
        observed after the request is in flight are effectively
        deferred to the next loop iteration where the result is
        suppressed before reaching the outbox.
        """
        self._aborted_ids.add(request_id)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _loop_once(self) -> None:
        # ----------------------------------------------------------
        # Three error domains, in order:
        # 1. recv / build_batch are pre-forward and recoverable. They
        #    synchronize through the TP CPU group before any model
        #    collective starts.
        # 2. encode_batch enters upstream TP collectives. A rank-local
        #    exception there is fatal to the stage group; do NOT try
        #    a post-hoc CPU gather while peers may still be blocked
        #    in NCCL.
        # 3. slice_results runs only on the entry rank after forward
        #    returned on every rank, so it can emit per-request
        #    errors and continue.
        # ----------------------------------------------------------
        messages, recv_err = self._recv_messages()
        if self._gather_pre_forward_error(recv_err):
            if self._is_entry_rank() and messages:
                self._emit_error(
                    messages,
                    (
                        recv_err
                        if recv_err is not None
                        else RuntimeError("peer-rank encoder recv failed")
                    ),
                )
            return
        if not messages:
            return

        plan = None
        build_err: BaseException | None = None
        try:
            plan = self.adapter.build_batch(messages)
        except Exception as exc:  # noqa: BLE001
            build_err = exc

        if self._gather_pre_forward_error(build_err):
            if self._is_entry_rank():
                self._emit_error(
                    messages,
                    (
                        build_err
                        if build_err is not None
                        else RuntimeError("peer-rank encoder build_batch failed")
                    ),
                )
            return

        try:
            raw = self.worker.encode_batch(plan)
        except Exception as exc:  # noqa: BLE001
            self._fatal_tp_forward_error(exc)
            raise  # unreachable — _fatal_tp_forward_error exits

        if not self._is_entry_rank():
            return

        try:
            results = self.adapter.slice_results(raw, plan, messages)
        except Exception as exc:  # noqa: BLE001
            logger.exception("EncoderScheduler.slice_results failed")
            self._emit_error(messages, exc)
            return

        for msg, out in zip(messages, results):
            if msg.request_id in self._aborted_ids:
                self._aborted_ids.discard(msg.request_id)
                continue
            self.outbox.put(
                OutgoingMessage(request_id=msg.request_id, type="result", data=out)
            )

    # ------------------------------------------------------------------
    # Recv / broadcast (two-channel: CPU group metadata, device tensors)
    # ------------------------------------------------------------------

    def _recv_messages(
        self,
    ) -> tuple[list[IncomingMessage], BaseException | None]:
        """Drain the inbox on the entry rank, broadcast to followers.

        Never raises — returns ``(messages, error)`` so a recv-time
        failure is treated as a recoverable pre-forward error. Always
        runs ``_strip_and_lift`` even at ``tp_size == 1`` because the
        default shm relay delivers CPU tensors and upstream
        ``get_*_feature`` only calls ``.type(dtype)``, not ``.to(device)``.
        """
        if self.worker.tp_size == 1:
            local, collect_err = self._collect_batch_or_error()
            if collect_err is not None or not local:
                return local, collect_err
            try:
                meta_msgs, tensor_lists, specs_lists = self._strip_and_lift(local)
            except Exception as exc:  # noqa: BLE001
                return local, exc
            return (
                self._reattach_lifted_tensors(meta_msgs, tensor_lists, specs_lists),
                None,
            )

        from sglang.srt.utils import broadcast_pyobj

        tp = self.worker.tp_group
        src_rank = self.worker.tp_rank if self._is_entry_rank() else 0

        if self._is_entry_rank():
            local, collect_err = self._collect_batch_or_error()
            if collect_err is not None:
                broadcast_pyobj(
                    [{"kind": _RECV_ERROR_KIND, "error": repr(collect_err)}],
                    self.worker.tp_rank,
                    tp.cpu_group,
                    src=src_rank,
                )
                return local, collect_err

            try:
                meta_msgs, tensor_lists, specs_lists = self._strip_and_lift(local)
            except Exception as exc:  # noqa: BLE001
                broadcast_pyobj(
                    [{"kind": _RECV_ERROR_KIND, "error": repr(exc)}],
                    self.worker.tp_rank,
                    tp.cpu_group,
                    src=src_rank,
                )
                return local, exc

            broadcast_pyobj(
                [meta_msgs, specs_lists],
                self.worker.tp_rank,
                tp.cpu_group,
                src=src_rank,
            )

            ok_flags = self._allocation_ready_gather(local_ok=True)
            if not all(ok_flags):
                return local, RuntimeError("peer-rank tensor allocation failed")

            import torch.distributed as dist

            for tensor_list in tensor_lists:
                for t in tensor_list:
                    dist.broadcast(t.contiguous(), src=src_rank, group=tp.device_group)
            return (
                self._reattach_lifted_tensors(meta_msgs, tensor_lists, specs_lists),
                None,
            )

        # Follower path
        payload = broadcast_pyobj([], self.worker.tp_rank, tp.cpu_group, src=src_rank)
        if (
            payload
            and isinstance(payload[0], dict)
            and payload[0].get("kind") == _RECV_ERROR_KIND
        ):
            return [], RuntimeError(
                f"entry rank failed before metadata broadcast: "
                f"{payload[0]['error']}"
            )
        meta_msgs, specs_lists = payload

        placeholders: list[list[torch.Tensor]] = []
        alloc_err: BaseException | None = None
        try:
            for specs in specs_lists:
                placeholders.append(
                    [
                        torch.empty(
                            spec.shape,
                            dtype=spec.dtype,
                            device=self.worker.device,
                        )
                        for spec in specs
                    ]
                )
        except Exception as exc:  # noqa: BLE001
            alloc_err = exc

        ok_flags = self._allocation_ready_gather(local_ok=alloc_err is None)
        if not all(ok_flags):
            return [], (
                alloc_err
                if alloc_err is not None
                else RuntimeError("peer-rank tensor allocation failed")
            )

        import torch.distributed as dist

        rebuilt: list[IncomingMessage] = []
        for meta_msg, specs, ph_list in zip(meta_msgs, specs_lists, placeholders):
            tensor_dict: dict[str, torch.Tensor] = {}
            for spec, t in zip(specs, ph_list):
                dist.broadcast(t, src=src_rank, group=tp.device_group)
                tensor_dict[spec.path] = t
            meta_msg.data.data = restore_tensors(meta_msg.data.data, tensor_dict)
            rebuilt.append(meta_msg)
        return rebuilt, None

    # ------------------------------------------------------------------
    # Cost-capped batch collection (entry rank)
    # ------------------------------------------------------------------

    def _next_message(self) -> IncomingMessage | None:
        if self._pending_messages:
            return self._pending_messages.popleft()
        try:
            return self.inbox.get(timeout=0.1)
        except _queue_mod.Empty:
            return None

    def _collect_batch_or_error(
        self,
    ) -> tuple[list[IncomingMessage], BaseException | None]:
        try:
            return self._collect_batch_from_inbox(), None
        except BatchCollectError as exc:
            return exc.messages, exc.error
        except Exception as exc:  # noqa: BLE001
            return [], exc

    def _collect_batch_from_inbox(self) -> list[IncomingMessage]:
        first = self._next_message()
        if first is None:
            return []
        if first.type != "new_request":
            return [first]

        batch = [first]
        try:
            batch_cost = self._message_cost(first)
        except Exception as exc:  # noqa: BLE001
            raise BatchCollectError(batch, exc) from exc

        deadline = time.monotonic() + self._max_batch_wait_s
        while len(batch) < self._max_batch_size:
            try:
                msg = self.inbox.get_nowait()
            except _queue_mod.Empty:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    msg = self.inbox.get(timeout=remaining)
                except _queue_mod.Empty:
                    break

            if msg.type != "new_request":
                self._pending_messages.append(msg)
                continue

            if self._max_batch_cost is not None:
                try:
                    cost = self._message_cost(msg)
                except Exception as exc:  # noqa: BLE001
                    batch.append(msg)
                    raise BatchCollectError(batch, exc) from exc
                if batch_cost + cost > self._max_batch_cost:
                    self._pending_messages.appendleft(msg)
                    break
                batch_cost += cost
            batch.append(msg)
        return batch

    def _message_cost(self, msg: IncomingMessage) -> int:
        if self._request_cost_fn is None or msg.type != "new_request":
            return 0
        return max(int(self._request_cost_fn(msg.data)), 0)

    # ------------------------------------------------------------------
    # Tensor lift / reattach (entry rank)
    # ------------------------------------------------------------------

    def _strip_and_lift(
        self,
        messages: list[IncomingMessage],
    ) -> tuple[
        list[IncomingMessage],
        list[list[torch.Tensor]],
        list[list[_TensorSpec]],
    ]:
        """Extract tensors out of each message's payload and stage on GPU.

        Returns three parallel lists:
        - ``meta_msgs``: the original messages with tensors removed from
          ``data.data`` (replaced by relay placeholders).
        - ``tensor_lists``: lifted tensors, one list per message, each
          tensor moved to ``self.worker.device``.
        - ``specs_lists``: typed ``_TensorSpec`` records carrying the
          real ``torch.dtype`` (the placeholder dict only has a string
          dtype, which won't survive a ``torch.empty(...)`` call on the
          follower).
        """
        meta_msgs: list[IncomingMessage] = []
        tensor_lists: list[list[torch.Tensor]] = []
        specs_lists: list[list[_TensorSpec]] = []

        for msg in messages:
            payload = msg.data
            data = getattr(payload, "data", payload)
            stripped, tensor_dict = extract_tensors(data)
            tensors: list[torch.Tensor] = []
            specs: list[_TensorSpec] = []
            for path, tensor in tensor_dict.items():
                staged = (
                    tensor
                    if tensor.device == self.worker.device
                    else tensor.to(self.worker.device, non_blocking=True)
                )
                tensors.append(staged)
                specs.append(
                    _TensorSpec(
                        path=path,
                        shape=tuple(staged.shape),
                        dtype=staged.dtype,
                    )
                )
            if hasattr(payload, "data"):
                payload.data = stripped
            else:
                msg.data = stripped
            meta_msgs.append(msg)
            tensor_lists.append(tensors)
            specs_lists.append(specs)
        return meta_msgs, tensor_lists, specs_lists

    def _reattach_lifted_tensors(
        self,
        meta_msgs: list[IncomingMessage],
        tensor_lists: list[list[torch.Tensor]],
        specs_lists: list[list[_TensorSpec]],
    ) -> list[IncomingMessage]:
        """Reverse of :meth:`_strip_and_lift` for the entry rank."""
        for msg, tensors, specs in zip(meta_msgs, tensor_lists, specs_lists):
            tensor_dict = {spec.path: t for spec, t in zip(specs, tensors)}
            payload = msg.data
            data = getattr(payload, "data", payload)
            restored = restore_tensors(data, tensor_dict)
            if hasattr(payload, "data"):
                payload.data = restored
            else:
                msg.data = restored
        return meta_msgs

    # ------------------------------------------------------------------
    # Pre-forward synchronization
    # ------------------------------------------------------------------

    def _gather_pre_forward_error(self, local_err: BaseException | None) -> bool:
        if self.worker.tp_size <= 1:
            return local_err is not None
        import torch.distributed as dist

        flags = [False] * self.worker.tp_size
        dist.all_gather_object(
            flags,
            local_err is not None,
            group=self.worker.tp_group.cpu_group,
        )
        return any(flags)

    def _allocation_ready_gather(self, *, local_ok: bool) -> list[bool]:
        if self.worker.tp_size <= 1:
            return [local_ok]
        import torch.distributed as dist

        flags = [False] * self.worker.tp_size
        dist.all_gather_object(flags, local_ok, group=self.worker.tp_group.cpu_group)
        return flags

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_entry_rank(self) -> bool:
        return getattr(self.worker, "is_entry_rank", self.worker.tp_rank == 0)

    def _emit_error(
        self,
        messages: list[IncomingMessage],
        error: BaseException,
    ) -> None:
        # Cheng's design note: SimpleScheduler exposes a single-request
        # ``_emit_error(request_id, error)`` helper, so reusing that
        # signature here would put a list[IncomingMessage] into
        # ``OutgoingMessage.request_id`` and break Stage's
        # ``request_id not in self._active_requests`` check. Iterate
        # explicitly per request instead.
        for msg in messages:
            self.outbox.put(
                OutgoingMessage(request_id=msg.request_id, type="error", data=error)
            )

    def _fatal_tp_forward_error(self, error: BaseException) -> None:
        """Exit non-zero after a TP forward fault.

        Once ``encode_batch`` has entered upstream SGLang TP
        collectives, a rank-local exception cannot recover safely:
        peers may be stuck in NCCL and never reach a CPU-side error
        gather. Force a child-process failure so the StageGroup
        monitor tears down the whole TP group and fails outstanding
        requests through the coordinator.
        """
        logger.exception("Fatal TP encoder forward failure: %s", error)
        os._exit(1)


__all__ = ["EncoderScheduler", "BatchCollectError"]
