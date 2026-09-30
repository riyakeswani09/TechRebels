ATTACK_STAGES = {
    "RECONNAISSANCE": 1,
    "INITIAL_ACCESS": 2,
    "LATERAL_MOVEMENT": 3,
    "COMMAND_AND_CONTROL": 4,
    "EXFILTRATION": 5
}


STAGE_NAMES = {
    1: "Reconnaissance",
    2: "Initial Access",
    3: "Lateral Movement",
    4: "Command & Control",
    5: "Exfiltration"
}


# BENIGN is not an attack stage.
# It represents normal network activity.
BENIGN_STAGE = "BENIGN"