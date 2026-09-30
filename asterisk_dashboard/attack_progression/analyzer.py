from stage_converter import convert_labels_to_stages
from trajectory import analyze_trajectory
from risk import assess_risk


def analyze_attack(labels):

    stages = convert_labels_to_stages(labels)

    if len(stages) < 2:
        return {
            "error": "Not enough valid attack stages for trajectory analysis"
        }

    transitions = analyze_trajectory(stages)

    current_stage = stages[-2]
    predicted_stage = stages[-1]

    risk_result = assess_risk(
        current_stage,
        predicted_stage
    )

    result = {
        "current_stage": current_stage,
        "predicted_stage": predicted_stage,
        "transitions": transitions,
        "progression_distance": risk_result["progression_distance"],
        "risk_level": risk_result["risk_level"]
    }

    return result