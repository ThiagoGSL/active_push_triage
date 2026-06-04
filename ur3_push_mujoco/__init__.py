import os 

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["UR3_PUSH_DATAPATH"] = os.path.join(project_root, "ur3_push_data")