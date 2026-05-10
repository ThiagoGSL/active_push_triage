import os 

init_path = os.path.dirname(__file__)
os.environ["PANDA_PUSH_DATAPATH"] = os.path.join(init_path[:init_path.find("precise_pushing") + len("precise_pushing")],"ur3_push_data")