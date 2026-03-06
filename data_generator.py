from __future__ import annotations

import random
import copy
from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict

from datasets import Dataset
from diplomacy import Game


STANDARD_POWERS = [
    "AUSTRIA",
    "ENGLAND",
    "FRANCE",
    "GERMANY",
    "ITALY",
    "RUSSIA",
    "TURKEY",
]
EXPECTED_FINAL_ANSWER = "DONE"


class RequiredInteractionSpec(TypedDict):
    participants: list[str]
    channel_type: Literal["dm", "group"]
    description: str


class OrderConstraintSpec(TypedDict):
    location: str
    canonical_order: str
    description: str


class TransitionTargetSpec(TypedDict, total=False):
    kind: Literal[
        "unit_at",
        "location_occupied_by",
        "move_succeeded",
        "move_failed",
        "unit_dislodged",
    ]
    power: str
    unit_type: str
    location: str
    from_location: str
    to_location: str


class RelevantActorObjectiveSpec(TypedDict):
    power: str
    contact_required: bool
    objective_text: str
    anchor_orders: list[str]
    reply_template: str


class DiplomacyRowInfo(TypedDict, total=False):
    session_id: str
    agent_name: str
    tracked_power: str
    contacts: list[str]
    contacts_map: dict[str, list[str]]
    task_prompt: str
    agent_prompt: str
    task_config: dict[str, Any]
    expected_answer: str
    initial_messages: list[dict[str, Any]]
    initial_phase: str
    initial_state: dict[str, Any]
    relevant_powers: list[str]
    actor_prompts: dict[str, str]
    required_interactions: list[RequiredInteractionSpec]
    order_constraints: list[OrderConstraintSpec]
    transition_target: TransitionTargetSpec | None
    relevant_actor_objectives: list[RelevantActorObjectiveSpec]
    deterministic_orders: dict[str, list[str]]
    environment_kind: Literal["tool_accuracy", "full_press"]
    session_deadline_epoch: float
    default_idle_sleep_seconds: float


@dataclass
class DiplomacySessionSpec:
    session_id: str
    environment_kind: Literal["tool_accuracy", "full_press"]
    tracked_power: str
    relevant_powers: list[str]
    contacts_map: dict[str, list[str]]
    initial_phase: str
    initial_state: dict[str, Any]
    task_prompt: str
    actor_prompts: dict[str, str]
    required_interactions: list[RequiredInteractionSpec]
    order_constraints: list[OrderConstraintSpec] = field(default_factory=list)
    transition_target: TransitionTargetSpec | None = None
    relevant_actor_objectives: list[RelevantActorObjectiveSpec] = field(default_factory=list)
    deterministic_orders: dict[str, list[str]] = field(default_factory=dict)
    task_config: dict[str, Any] = field(default_factory=dict)
    initial_messages: list[dict[str, Any]] = field(default_factory=list)
    expected_answer: str = EXPECTED_FINAL_ANSWER


def _default_contacts_map() -> dict[str, list[str]]:
    return {
        power: [other for other in STANDARD_POWERS if other != power]
        for power in STANDARD_POWERS
    }


def _order_location(order: str) -> str:
    parts = order.split()
    if len(parts) < 2:
        raise ValueError(f"Unable to parse order location from {order!r}")
    return parts[1]


def _order_destination(order: str) -> str | None:
    parts = order.split()
    if "-" not in parts:
        return None
    dash_index = parts.index("-")
    if dash_index + 1 >= len(parts):
        return None
    return parts[dash_index + 1]


def _movement_phase_name(phase_name: str) -> bool:
    return len(phase_name) >= 6 and phase_name.endswith("M") and phase_name[0] in {"S", "F"}


def _holds_first(options: list[str]) -> list[str]:
    return sorted(options, key=lambda order: (0 if order.endswith(" H") else 1, order))


def _sample_complete_orders(
    game: Game,
    power: str,
    rng: random.Random,
    *,
    fixed_orders: dict[str, str] | None = None,
    prefer_holds: bool = False,
) -> list[str]:
    fixed_orders = fixed_orders or {}
    legal_orders = game.get_all_possible_orders()
    orderable_locations = list(game.get_orderable_locations(power))
    selected: list[str] = []
    for location in orderable_locations:
        if location in fixed_orders:
            selected.append(fixed_orders[location])
            continue
        options = list(legal_orders.get(location, []))
        if not options:
            continue
        if prefer_holds:
            options = _holds_first(options)
            selected.append(options[0])
        else:
            selected.append(rng.choice(options))
    return selected


def _clone_game(game: Game) -> Game:
    clone = Game()
    clone.set_state(game.get_state())
    return clone


def _sample_reachable_movement_game(rng: random.Random) -> Game:
    game = Game()
    if rng.random() < 0.5:
        return game
    for power in STANDARD_POWERS:
        game.set_orders(power, _sample_complete_orders(game, power, rng))
    game.process()
    return game


def _tool_accuracy_reply_template(required_order: str) -> str:
    return (
        "REQUIRED_ORDER:\n"
        f"- {required_order}\n"
        "COMMITMENT:\n"
        "- I have disclosed the required tracked-power order."
    )


def _build_tool_accuracy_actor_prompt(
    *,
    tracked_power: str,
    counterpart: str,
    required_order: str,
) -> str:
    reply_template = _tool_accuracy_reply_template(required_order)
    return (
        f"You are {counterpart}.\n"
        f"Your hidden objective this phase is to help {tracked_power} complete its tool-accuracy protocol.\n"
        f"If {tracked_power} messages you directly, reply exactly with:\n{reply_template}\n"
        "Do not proactively message first.\n"
        "If you submit orders, prefer holding your units.\n"
        "Finalize with DONE after replying and, if you choose to submit, after submitting a complete legal order set."
    )


def _build_tool_accuracy_task_prompt(
    *,
    tracked_power: str,
    counterpart: str,
    required_order: str,
) -> str:
    return (
        f"You are {tracked_power}.\n"
        f"This is a tool-accuracy Diplomacy movement-phase task.\n"
        f"Protocol requirements:\n"
        f"- DM {counterpart} and read the reply after it arrives.\n"
        "- Stay within the invalid-tool-call budget of 3 rejected tool calls.\n"
        "- Submit a complete legal order set for all of your current orderable units.\n"
        f"- Ensure your final submission includes this canonical order exactly:\n"
        f"  {required_order}\n"
        "- Finish with DONE only after submitting orders."
    )


def _build_full_press_reply_template(
    *,
    objective_text: str,
    anchor_orders: list[str],
) -> str:
    lines = [
        "OBJECTIVE:",
        f"- {objective_text}",
        "ANCHOR_ORDERS:",
    ]
    if anchor_orders:
        lines.extend(f"- {order}" for order in anchor_orders)
    else:
        lines.append("- NONE")
    lines.extend(
        [
            "COMMITMENT:",
            "- I will pursue this objective if it remains legal.",
        ]
    )
    return "\n".join(lines)


def _build_full_press_actor_prompt(
    *,
    tracked_power: str,
    objective: RelevantActorObjectiveSpec,
) -> str:
    contact_required = "YES" if objective["contact_required"] else "NO"
    anchor_orders = "\n".join(f"- {order}" for order in objective["anchor_orders"]) or "- NONE"
    return (
        f"You are {objective['power']}.\n"
        f"Your hidden objective this phase is:\n{objective['objective_text']}\n"
        f"CONTACT_REQUIRED: {contact_required}\n"
        f"ANCHOR_ORDERS:\n{anchor_orders}\n"
        "Messaging policy:\n"
        f"- If {tracked_power} messages you, reply exactly with:\n{objective['reply_template']}\n"
        f"- If CONTACT_REQUIRED is YES, do not reveal or commit before {tracked_power} contacts you.\n"
        "Submission policy:\n"
        "- Prioritize the anchor orders if they remain legal.\n"
        "- Hold your other units unless an anchor order becomes illegal.\n"
        "- Submit a complete legal order set, then finalize with DONE."
    )


def _build_full_press_task_prompt(
    *,
    tracked_power: str,
    counterpart: str,
    target: TransitionTargetSpec,
) -> str:
    if target["kind"] == "unit_at":
        outcome = (
            f"end the movement phase with your {target['unit_type']} in {target['location']}"
        )
    elif target["kind"] == "location_occupied_by":
        outcome = f"end the movement phase with {tracked_power} occupying {target['location']}"
    elif target["kind"] == "move_succeeded":
        outcome = (
            f"make the move from {target['from_location']} to {target['to_location']} succeed"
        )
    elif target["kind"] == "move_failed":
        outcome = (
            f"ensure the move from {target['from_location']} to {target['to_location']} fails"
        )
    else:
        outcome = f"dislodge the unit at {target['location']}"
    return (
        f"You are {tracked_power}.\n"
        "This is a full-press Diplomacy movement-phase task.\n"
        f"Your objective is to {outcome}.\n"
        f"You must coordinate with {counterpart}: DM them, read their reply after it arrives, "
        "and then submit a complete legal order set.\n"
        "Stay within the invalid-tool-call budget of 3 rejected tool calls.\n"
        "Success depends on the actual post-adjudication phase result.\n"
        "Finish with DONE only after submitting orders."
    )


def _build_row(session: DiplomacySessionSpec) -> dict[str, Any]:
    tracked_contacts = list(session.contacts_map[session.tracked_power])
    info: DiplomacyRowInfo = {
        "session_id": session.session_id,
        "agent_name": session.tracked_power,
        "tracked_power": session.tracked_power,
        "contacts": tracked_contacts,
        "contacts_map": {k: list(v) for k, v in session.contacts_map.items()},
        "task_prompt": session.task_prompt,
        "agent_prompt": session.task_prompt,
        "task_config": dict(session.task_config),
        "expected_answer": session.expected_answer,
        "initial_messages": list(session.initial_messages),
        "initial_phase": session.initial_phase,
        "initial_state": copy.deepcopy(session.initial_state),
        "relevant_powers": list(session.relevant_powers),
        "actor_prompts": dict(session.actor_prompts),
        "required_interactions": list(session.required_interactions),
        "order_constraints": list(session.order_constraints),
        "transition_target": dict(session.transition_target) if session.transition_target else None,
        "relevant_actor_objectives": list(session.relevant_actor_objectives),
        "deterministic_orders": {k: list(v) for k, v in session.deterministic_orders.items()},
        "environment_kind": session.environment_kind,
    }
    return {
        "example_id": session.session_id,
        "prompt": [{"role": "user", "content": session.task_prompt}],
        "answer": session.expected_answer,
        "info": info,
        "task": f"diplomacy-{session.environment_kind}",
    }


def _initial_required_interactions(counterpart: str) -> list[RequiredInteractionSpec]:
    return [
        {
            "participants": [counterpart],
            "channel_type": "dm",
            "description": f"Tracked power must DM {counterpart}.",
        },
        {
            "participants": [counterpart],
            "channel_type": "dm",
            "description": f"Tracked power must read {counterpart}'s reply.",
        },
    ]


def _tool_accuracy_session(idx: int, rng: random.Random) -> DiplomacySessionSpec:
    game = _sample_reachable_movement_game(rng)
    phase_name = game.get_current_phase()
    if not _movement_phase_name(phase_name):
        raise ValueError(f"Expected movement phase, got {phase_name}")
    orderable = game.get_orderable_locations()
    tracked_candidates = [power for power, locations in orderable.items() if locations]
    tracked_power = rng.choice(tracked_candidates)
    counterpart = rng.choice([power for power in STANDARD_POWERS if power != tracked_power])
    constrained_location = rng.choice(list(orderable[tracked_power]))
    constrained_order = rng.choice(list(game.get_all_possible_orders()[constrained_location]))
    required_interactions = _initial_required_interactions(counterpart)
    order_constraints: list[OrderConstraintSpec] = [
        {
            "location": constrained_location,
            "canonical_order": constrained_order,
            "description": f"{tracked_power} must submit {constrained_order}.",
        }
    ]
    actor_prompt = _build_tool_accuracy_actor_prompt(
        tracked_power=tracked_power,
        counterpart=counterpart,
        required_order=constrained_order,
    )
    return DiplomacySessionSpec(
        session_id=f"tool_accuracy_{idx:05d}_{rng.randint(1000, 9999)}",
        environment_kind="tool_accuracy",
        tracked_power=tracked_power,
        relevant_powers=[counterpart],
        contacts_map=_default_contacts_map(),
        initial_phase=phase_name,
        initial_state=copy.deepcopy(game.get_state()),
        task_prompt=_build_tool_accuracy_task_prompt(
            tracked_power=tracked_power,
            counterpart=counterpart,
            required_order=constrained_order,
        ),
        actor_prompts={counterpart: actor_prompt},
        required_interactions=required_interactions,
        order_constraints=order_constraints,
        relevant_actor_objectives=[
            {
                "power": counterpart,
                "contact_required": True,
                "objective_text": f"Disclose the required {tracked_power} order when contacted.",
                "anchor_orders": [constrained_order],
                "reply_template": _tool_accuracy_reply_template(constrained_order),
            }
        ],
        task_config={
            "tracked_power": tracked_power,
            "counterpart_power": counterpart,
            "phase_name": phase_name,
        },
    )


def _find_support_candidate(
    game: Game,
    *,
    tracked_power: str,
    counterpart: str,
    rng: random.Random,
) -> tuple[str, str, str, str, str] | None:
    legal_orders = game.get_all_possible_orders()
    tracked_locations = list(game.get_orderable_locations(tracked_power))
    counterpart_locations = list(game.get_orderable_locations(counterpart))
    rng.shuffle(tracked_locations)
    rng.shuffle(counterpart_locations)
    for tracked_location in tracked_locations:
        tracked_options = list(legal_orders.get(tracked_location, []))
        rng.shuffle(tracked_options)
        for tracked_order in tracked_options:
            destination = _order_destination(tracked_order)
            if destination is None:
                continue
            support_signature = f"S {tracked_order}"
            for counterpart_location in counterpart_locations:
                counterpart_options = list(legal_orders.get(counterpart_location, []))
                rng.shuffle(counterpart_options)
                for counterpart_order in counterpart_options:
                    if support_signature in counterpart_order:
                        unit_type = tracked_order.split()[0]
                        return tracked_location, tracked_order, counterpart_location, counterpart_order, unit_type
    return None


def _full_press_session(idx: int, rng: random.Random) -> DiplomacySessionSpec:
    for _ in range(200):
        game = _sample_reachable_movement_game(rng)
        phase_name = game.get_current_phase()
        if not _movement_phase_name(phase_name):
            continue
        orderable = game.get_orderable_locations()
        tracked_candidates = [power for power, locations in orderable.items() if locations]
        if len(tracked_candidates) < 2:
            continue
        rng.shuffle(tracked_candidates)
        for tracked_power in tracked_candidates:
            counterpart_candidates = [power for power in STANDARD_POWERS if power != tracked_power and orderable.get(power)]
            rng.shuffle(counterpart_candidates)
            for counterpart in counterpart_candidates:
                candidate = _find_support_candidate(
                    game,
                    tracked_power=tracked_power,
                    counterpart=counterpart,
                    rng=rng,
                )
                if candidate is None:
                    continue
                (
                    tracked_location,
                    tracked_anchor_order,
                    counterpart_location,
                    counterpart_anchor_order,
                    tracked_unit_type,
                ) = candidate
                non_relevant_powers = [
                    power
                    for power in STANDARD_POWERS
                    if power not in {tracked_power, counterpart}
                ]
                deterministic_orders = {
                    power: _sample_complete_orders(game, power, rng)
                    for power in non_relevant_powers
                }
                tracked_orders = _sample_complete_orders(
                    game,
                    tracked_power,
                    rng,
                    fixed_orders={tracked_location: tracked_anchor_order},
                )
                counterpart_orders = _sample_complete_orders(
                    game,
                    counterpart,
                    rng,
                    fixed_orders={counterpart_location: counterpart_anchor_order},
                    prefer_holds=True,
                )
                adjudication_game = _clone_game(game)
                for power, orders in deterministic_orders.items():
                    adjudication_game.set_orders(power, orders)
                adjudication_game.set_orders(tracked_power, tracked_orders)
                adjudication_game.set_orders(counterpart, counterpart_orders)
                adjudication_game.process()
                destination = _order_destination(tracked_anchor_order)
                if destination is None:
                    continue
                tracked_units = adjudication_game.get_units(tracked_power)
                target_order = f"{tracked_unit_type} {destination}"
                if target_order not in tracked_units:
                    continue
                target_kind = rng.choice(["unit_at", "location_occupied_by"])
                if target_kind == "unit_at":
                    transition_target: TransitionTargetSpec = {
                        "kind": "unit_at",
                        "power": tracked_power,
                        "unit_type": tracked_unit_type,
                        "location": destination,
                    }
                else:
                    transition_target = {
                        "kind": "location_occupied_by",
                        "power": tracked_power,
                        "location": destination,
                    }
                contact_required = rng.random() < 0.5
                objective_text = (
                    f"Coordinate with {tracked_power} so their unit from {tracked_location} ends the phase in "
                    f"{destination}. Submit {counterpart_anchor_order} if legal and hold your other units."
                )
                reply_template = _build_full_press_reply_template(
                    objective_text=objective_text,
                    anchor_orders=[counterpart_anchor_order],
                )
                actor_objective: RelevantActorObjectiveSpec = {
                    "power": counterpart,
                    "contact_required": contact_required,
                    "objective_text": objective_text,
                    "anchor_orders": [counterpart_anchor_order],
                    "reply_template": reply_template,
                }
                return DiplomacySessionSpec(
                    session_id=f"full_press_{idx:05d}_{rng.randint(1000, 9999)}",
                    environment_kind="full_press",
                    tracked_power=tracked_power,
                    relevant_powers=[counterpart],
                    contacts_map=_default_contacts_map(),
                    initial_phase=phase_name,
                    initial_state=copy.deepcopy(game.get_state()),
                    task_prompt=_build_full_press_task_prompt(
                        tracked_power=tracked_power,
                        counterpart=counterpart,
                        target=transition_target,
                    ),
                    actor_prompts={
                        counterpart: _build_full_press_actor_prompt(
                            tracked_power=tracked_power,
                            objective=actor_objective,
                        )
                    },
                    required_interactions=_initial_required_interactions(counterpart),
                    transition_target=transition_target,
                    relevant_actor_objectives=[actor_objective],
                    deterministic_orders=deterministic_orders,
                    task_config={
                        "tracked_power": tracked_power,
                        "counterpart_power": counterpart,
                        "phase_name": phase_name,
                        "tracked_anchor_order": tracked_anchor_order,
                        "counterpart_anchor_order": counterpart_anchor_order,
                        "planned_tracked_orders": list(tracked_orders),
                        "planned_counterpart_orders": list(counterpart_orders),
                    },
                )
    raise RuntimeError("Unable to generate a full-press session with a supported target.")


def build_tool_accuracy_sessions(
    *,
    num_sessions: int,
    seed: int | None = None,
) -> list[DiplomacySessionSpec]:
    rng = random.Random(seed)
    return [_tool_accuracy_session(idx, rng) for idx in range(num_sessions)]


def build_full_press_sessions(
    *,
    num_sessions: int,
    seed: int | None = None,
) -> list[DiplomacySessionSpec]:
    rng = random.Random(seed)
    return [_full_press_session(idx, rng) for idx in range(num_sessions)]


def build_tool_accuracy_dataset(
    *,
    num_sessions: int,
    seed: int | None = None,
) -> Dataset:
    return Dataset.from_list(
        [_build_row(session) for session in build_tool_accuracy_sessions(num_sessions=num_sessions, seed=seed)]
    )


def build_full_press_dataset(
    *,
    num_sessions: int,
    seed: int | None = None,
) -> Dataset:
    return Dataset.from_list(
        [_build_row(session) for session in build_full_press_sessions(num_sessions=num_sessions, seed=seed)]
    )
