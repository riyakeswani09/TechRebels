from stages import ATTACK_STAGES, BENIGN_STAGE


def check_transition(current_stage, next_stage):

    # Both states are normal network activity
    if current_stage == BENIGN_STAGE and next_stage == BENIGN_STAGE:
        return "NO_ATTACK"

    # Attack starts after normal activity
    if current_stage == BENIGN_STAGE and next_stage not in [BENIGN_STAGE, "UNKNOWN"]:
        return "ATTACK_START"

    # Attack returns to normal activity
    if current_stage not in [BENIGN_STAGE, "UNKNOWN"] and next_stage == BENIGN_STAGE:
        return "ATTACK_ENDED_OR_UNCERTAIN"

    # Unknown state cannot be reliably compared
    if current_stage == "UNKNOWN" or next_stage == "UNKNOWN":
        return "UNKNOWN_TRANSITION"

    # Normal attack-stage progression
    current = ATTACK_STAGES[current_stage]
    next_state = ATTACK_STAGES[next_stage]

    if next_state > current:
        return "FORWARD_PROGRESSION"

    elif next_state == current:
        return "SAME_STAGE"

    else:
        return "BACKWARD_OR_UNCERTAIN"