import os


PROJECT_NAME = 'mt-sepe-sr-mbrl'
ENTITY_NAME = 'riccardos'

PROJECT_PATH = os.getcwd()
while os.path.basename(PROJECT_PATH) != "Physics_Informed_Model_Based_RL":
    PROJECT_PATH = os.path.dirname(PROJECT_PATH)


__all__ = ["PROJECT_PATH", "PROJECT_NAME", "ENTITY_NAME"]
