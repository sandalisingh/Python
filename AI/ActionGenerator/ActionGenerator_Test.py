import numpy as np
from States import ActionStates, Range, PersonalityIndex, EmotionStates
from ActionGenerator import ActionGenerator

# Instantiate ActionGenerator
action_gen = ActionGenerator()

# Sample input data
personality_vector = [6, 4, 8, 5, 7]

while(True):
    emotional_state = EmotionStates[input("Emotion state : ")]
    previous_action_state = ActionStates[input("Previous action state : ")]
    emotional_state = EmotionStates["Joy"]
    previous_action_state = ActionStates["Attacking"]
    print(f"Current state: (Emotion state = {emotional_state}, Action state = {previous_action_state})")
    current_state = (emotional_state, previous_action_state)

    # Call action_generator
    action_index = action_gen.action_generator(personality_vector, emotional_state, previous_action_state)
    print("-> Generated Action : ", ActionStates.index_to_enum(action_index).name)

    # Call q_learning
    action_gen.q_learning(personality_vector, current_state, action_index)


