import numpy as np
from States import ActionStates, Range, PersonalityIndex, EmotionStates, get_action
from ActionGenerator import ActionGenerator

# Instantiate ActionGenerator
action_gen = ActionGenerator()

# Sample input data
personality_vector = [6, 4, 8, 5, 7]

while(True):
    emotional_state_index = EmotionStates[input("Emotion state : ")]
    previous_action_state_index = ActionStates[input("Previous action state : ")]
    current_state = (emotional_state_index, previous_action_state_index)

    # Call action_generator
    action_index = action_gen.action_generator(personality_vector, emotional_state_index, previous_action_state_index)
    print("\n\n-> Generated Action : ", get_action(action_index))
    print("\n\n")

    # Call q_learning
    action_gen.q_learning(personality_vector, current_state, action_index)


