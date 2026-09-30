from stages import ATTACK_STAGES, BENIGN_STAGE


def calculate_progression_distance(current_stage, predicted_stage):

    # Cannot calculate attack-stage distance involving BENIGN
    if current_stage == BENIGN_STAGE or predicted_stage == BENIGN_STAGE:
        return None

    # Cannot calculate distance involving UNKNOWN
    if current_stage == "UNKNOWN" or predicted_stage == "UNKNOWN":
        return None

    current = ATTACK_STAGES[current_stage]
    predicted = ATTACK_STAGES[predicted_stage]

    distance = predicted - current

    return distance


def assess_risk(current_stage, predicted_stage):

    distance = calculate_progression_distance(
        current_stage,
        predicted_stage
    )

    # No reliable attack-stage distance
    if distance is None:

        if predicted_stage == BENIGN_STAGE:
            risk_level = "LOW"

        elif predicted_stage == "UNKNOWN":
            risk_level = "UNKNOWN"

        else:
            risk_level = "UNDETERMINED"

    elif distance <= 0:
        risk_level = "LOW"

    elif distance == 1:
        risk_level = "MEDIUM"

    else:
        risk_level = "HIGH"

    return {
        "current_stage": current_stage,
        "predicted_stage": predicted_stage,
        "progression_distance": distance,
        "risk_level": risk_level
    }