from progression import check_transition
from trajectory import analyze_trajectory
from label_mapping import LABEL_TO_STAGE

# --------------------------------
# Test 1: Single transition tests
# --------------------------------

result1 = check_transition("RECONNAISSANCE", "INITIAL_ACCESS")
print("Test 1:", result1)

result2 = check_transition("RECONNAISSANCE", "RECONNAISSANCE")
print("Test 2:", result2)

result3 = check_transition("INITIAL_ACCESS", "RECONNAISSANCE")
print("Test 3:", result3)


# --------------------------------
# Test 4: Complete attack trajectory
# --------------------------------

attack_trajectory = [
    "RECONNAISSANCE",
    "RECONNAISSANCE",
    "INITIAL_ACCESS",
    "INITIAL_ACCESS",
    "LATERAL_MOVEMENT"
]

result4 = analyze_trajectory(attack_trajectory)

print("Test 4:", result4)

from risk import calculate_progression_distance


# --------------------------------
# Test 5: Progression distance
# --------------------------------

distance1 = calculate_progression_distance(
    "RECONNAISSANCE",
    "LATERAL_MOVEMENT"
)

print("Test 5:", distance1)


distance2 = calculate_progression_distance(
    "INITIAL_ACCESS",
    "RECONNAISSANCE"
)

print("Test 6:", distance2)

from risk import assess_risk


# --------------------------------
# Test 7: Risk assessment
# --------------------------------

risk1 = assess_risk(
    "RECONNAISSANCE",
    "INITIAL_ACCESS"
)

print("Test 7:", risk1)


risk2 = assess_risk(
    "RECONNAISSANCE",
    "LATERAL_MOVEMENT"
)

print("Test 8:", risk2)


risk3 = assess_risk(
    "INITIAL_ACCESS",
    "RECONNAISSANCE"
)

print("Test 9:", risk3)

from analyzer import analyze_attack


# Test 10: End-to-end analysis with CIC-IDS2017 labels

attack_trajectory = [
    "BENIGN",
    "PortScan",
    "PortScan",
    "FTP-Patator",
    "FTP-Patator",
    "Bot"
]

analysis_result = analyze_attack(attack_trajectory)

print("Test 10:", analysis_result)

# --------------------------------
# Test 11: CIC-IDS2017 label mapping
# --------------------------------

test_labels = [
    "PortScan",
    "FTP-Patator",
    "SSH-Patator",
    "Web Attack - Brute Force",
    "Web Attack - XSS",
    "Web Attack - Sql Injection",
    "Heartbleed",
    "Bot"
]

for label in test_labels:

    stage = LABEL_TO_STAGE[label]

    print(label, "→", stage)

    from stage_converter import convert_labels_to_stages


# --------------------------------
# Test 12: Convert dataset labels
# to attack stages
# --------------------------------

window_labels = [
    "PortScan",
    "PortScan",
    "FTP-Patator",
    "FTP-Patator",
    "Bot"
]

converted_stages = convert_labels_to_stages(window_labels)

print("Test 12:", converted_stages)

from analyzer import analyze_attack


# --------------------------------
# Test 13: Complete CIC-IDS2017
# attack progression analysis
# --------------------------------

cic_ids_trajectory = [
    "PortScan",
    "PortScan",
    "FTP-Patator",
    "FTP-Patator",
    "Bot"
]

final_result = analyze_attack(cic_ids_trajectory)

print("Test 13:", final_result)