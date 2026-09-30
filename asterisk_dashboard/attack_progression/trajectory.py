from progression import check_transition


def analyze_trajectory(stages):

    transitions = []

    for i in range(len(stages) - 1):

        current_stage = stages[i]
        next_stage = stages[i + 1]

        result = check_transition(current_stage, next_stage)

        transitions.append(result)

    return transitions