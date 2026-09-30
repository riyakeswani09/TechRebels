from stages import ATTACK_STAGES, BENIGN_STAGE


LABEL_TO_STAGE = {

    # Normal network activity
    "BENIGN": BENIGN_STAGE,

    # Reconnaissance
    "PortScan": "RECONNAISSANCE",

    # Initial Access
    "FTP-Patator": "INITIAL_ACCESS",
    "SSH-Patator": "INITIAL_ACCESS",
    "Web Attack - Brute Force": "INITIAL_ACCESS",
    "Web Attack - XSS": "INITIAL_ACCESS",
    "Web Attack - Sql Injection": "INITIAL_ACCESS",
    "Heartbleed": "INITIAL_ACCESS",

    # Command and Control
    "Bot": "COMMAND_AND_CONTROL"
}