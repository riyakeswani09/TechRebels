from label_mapping import LABEL_TO_STAGE
from stages import BENIGN_STAGE


def convert_labels_to_stages(labels):

    stages = []

    for label in labels:

        if label in LABEL_TO_STAGE:
            stage = LABEL_TO_STAGE[label]
            stages.append(stage)

        else:
            # Keep unknown labels instead of silently removing them
            stages.append("UNKNOWN")

    return stages