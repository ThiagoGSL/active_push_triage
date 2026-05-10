import os 

init_path = os.path.dirname(__file__)
os.environ["PANDA_PUSH_DATAPATH"] = os.path.join(init_path[:init_path.find("ur3_push") + len("ur3_push")],"ur3_push_data")