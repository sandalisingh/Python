import numpy as np
from States import logging, ActionStates, Range, PersonalityIndex, EmotionStates
from prettytable import PrettyTable
import math

class ActionGenerator:

    def __init__(self):
        # Initialize Q-table
        self.no_of_personality_states = 5    # OCEAN personality model
        self.no_of_ranges_of_personality_states = 3  # 3 ranges for each personality states (0-3, 4-7, 8-10)
        self.no_of_emotional_states = 28  # 10 emotional states
        self.no_of_action_states = 10  # 10 action states

        self.load_q_table()

        # Parameters
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.9  # Discount factor
        self.epsilon = 0.3  # Exploration rate

    #   Q TABLE

    def initialize_q_table(self):
        # initialize Q table
        # 5 ocean personality * 3 ranges (0-3,4-7,8-10) * 28 emotional states * 10 FROM action states * 10 TO action states
        return np.random.random((self.no_of_personality_states, self.no_of_ranges_of_personality_states,
                                 self.no_of_emotional_states, self.no_of_action_states, self.no_of_action_states))

    def save_q_table(self):
        try:
            np.save('Action_Q_Table.npy', self.Q)
            logging("info", "Q-table saved successfully.")
        except Exception as e:
            logging("error", str(e))
        
    def load_q_table(self):
        try:
            self.Q = np.load('Action_Q_Table.npy')
            logging("info", "Q-table loaded successfully.")
        except Exception as e:
            logging("error", str(e))
            self.Q = self.initialize_q_table()
            logging("info", "Initialized new Q-table.")

    def print_q_table(self, q_table, emotion_index, prev_action_index, personality=None, title="Q table"):
        action_names = [ActionStates.get_action(i) for i in range(self.no_of_action_states)]
        headers = ["Prev Action"] + list(action_names)

        table = PrettyTable(headers)
        table.maxwidth = 80

        prev_action_name = ActionStates.get_action(prev_action_index)
        
        # Round Q-values and create row
        rounded_q_values = [round(val, 2) for val in q_table]  # Round to 2 decimal places
        row = [prev_action_name] + rounded_q_values
        table.add_row(row)

        # Print the formatted Q-table
        table.title = title + f" ({PersonalityIndex.get_personality(personality)}) ({EmotionStates.index_to_enum(emotion_index)})"
        print(table)  

    #   ACTION GENERATION

    def action_generator(self, personality_vector, emotional_state_index, previous_action_state_index) :
        # Choose action based on epsilon-greedy policy
        if np.random.rand() < self.epsilon: 
            logging("info", "Exploration")
            final_action_state_index = np.random.randint(self.no_of_action_states)  # Choose random action
            print("\nRandom final action state index = ", final_action_state_index)
        else:
            logging("info", "Exploitation")

            # Initialize an array for 10 action states with value 1
            q_val_array = [1] * self.no_of_action_states

            # Calculate combined Q-values for each action state
            for i in range(self.no_of_personality_states):  # Iterate over 5 ocean personalities
                personality_index = Range.index_to_range_value(personality_vector[i])
                extracted_q_table = self.Q[i][personality_index][emotional_state_index.value][previous_action_state_index.value]
                self.print_q_table(extracted_q_table.tolist(), emotional_state_index.value, previous_action_state_index.value, i)
                for j in range(self.no_of_action_states):  # Iterate over 10 action states
                    q_val_array[j] *= extracted_q_table[j]

            # Print the q_val_array using the print_q_table function
            self.print_q_table(q_val_array, emotional_state_index.value, previous_action_state_index.value, None, "Multiplied Q table")  # Reshape for a single row

            # Select action state
            final_action_state_index = np.argmax(q_val_array)
            print("\nFinal action state index = ", final_action_state_index)

        print("\n")
        return final_action_state_index

    #   REINFORCEMENT LEARNING

    def calculate_reward(self, personality_vector, current_action, next_action):
        transition_reward = 0
        preference_reward = 0
        reward = 0
        
        # Define all possible transitions and their associated rewards
        transitions = {
            # Unfavorable transitions
            (ActionStates.Patrolling, ActionStates.Attacking): -1,
            (ActionStates.Attacking, ActionStates.Fleeing): -1,
            (ActionStates.Celebrating, ActionStates.Resting): -1,
            (ActionStates.Helping, ActionStates.Attacking): -1,
            (ActionStates.Following, ActionStates.Fleeing): -1,
            # Favorable transitions
            (ActionStates.Interacting, ActionStates.Helping): 1,
            (ActionStates.Interacting, ActionStates.Celebrating): 1,
            (ActionStates.Fleeing, ActionStates.Resting): 1,
            (ActionStates.Fleeing, ActionStates.Interacting): 1,
            (ActionStates.Searching, ActionStates.Interacting): 1,
            (ActionStates.Searching, ActionStates.Patrolling): 1,
            (ActionStates.Attacking, ActionStates.Resting): 1,
            (ActionStates.Attacking, ActionStates.Interacting): 1,
            (ActionStates.Attacking, ActionStates.Celebrating): 1
            # Add more transitions as needed
        }

        # Check if the current and next actions correspond to a defined transition
        transition_key = (ActionStates[ActionStates.get_action(current_action)], ActionStates[ActionStates.get_action(next_action)])
        if transition_key in transitions:
            transition_reward = transitions[transition_key]

        print("\nTransition Reward = ", transition_reward)

        # preferable action states for each personality
        personality_preferences = {
            PersonalityIndex.Openness.value: {
                #   imaginative, curious, open-minded, 
                Range.High: (ActionStates.Helping, ActionStates.Alerted, ActionStates.Interacting, ActionStates.Celebrating),
                Range.Medium: (ActionStates.Patrolling, ActionStates.Searching, ActionStates.Interacting),
                #   practical, and prefer familiar routines
                Range.Low: (ActionStates.Attacking, ActionStates.Resting, ActionStates.Following, ActionStates.Patrolling),
            },
            PersonalityIndex.Conscientiousness.value: {     
                # organized, responsible, dependable, and goal-oriented
                Range.High: (ActionStates.Attacking, ActionStates.Patrolling, ActionStates.Following, ActionStates.Helping),
                Range.Medium: (ActionStates.Interacting, ActionStates.Searching, ActionStates.Alerted, ActionStates.Patrolling),
                #   disorganized, careless, unreliable, and impulsive
                Range.Low: (ActionStates.Fleeing, ActionStates.Resting, ActionStates.Celebrating, ActionStates.Interacting),
            },
            PersonalityIndex.Extraversion.value: {
                #   sociability, assertiveness, talkativeness, outgoing, energetic
                Range.High: (ActionStates.Helping, ActionStates.Attacking, ActionStates.Interacting, ActionStates.Celebrating, ActionStates.Following),
                Range.Medium: (ActionStates.Patrolling, ActionStates.Alerted, ActionStates.Interacting),
                #   introverts, more reserved, reflective, prefer solitary activities
                Range.Low: (ActionStates.Fleeing, ActionStates.Resting, ActionStates.Searching, ActionStates.Patrolling),
            },
            PersonalityIndex.Agreeableness.value: {
                #   cooperative, empathetic, considerate, compassionate, trusting, and accommodating
                Range.High: (ActionStates.Helping, ActionStates.Interacting, ActionStates.Celebrating, ActionStates.Following),
                Range.Medium: (ActionStates.Interacting, ActionStates.Following, ActionStates.Patrolling),
                #   competitiveness, skepticism, less willing to compromise or cooperate
                Range.Low: (ActionStates.Resting, ActionStates.Fleeing, ActionStates.Attacking, ActionStates.Searching),
            },
            PersonalityIndex.Neuroticism.value: {
                #   anxiety, depression, moodiness, vulnerability to stress, emotional instability
                Range.High: (ActionStates.Resting, ActionStates.Attacking, ActionStates.Fleeing, ActionStates.Interacting),
                Range.Medium: (ActionStates.Alerted, ActionStates.Searching, ActionStates.Following, ActionStates.Interacting),
                #   emotionally stable, resilient, calmness, even-temperedness
                Range.Low: (ActionStates.Helping, ActionStates.Interacting, ActionStates.Patrolling, ActionStates.Celebrating),
            },
        }

        # Check if next_action is one of the preferred actions for each personality trait
        for trait, preferences in personality_preferences.items():
            index = Range.index_to_range_key(personality_vector[trait])
            if ActionStates[ActionStates.get_action(next_action)] in preferences[index]:
                preference_reward += 1

        print("Preference Reward = ", preference_reward)

        reward = transition_reward + preference_reward
        
        # Normalize the total reward 
        max_reward = max(reward for reward in transitions.values()) + len(personality_preferences)  # Assuming max reward for preference is 1 per trait
        min_reward = min(reward for reward in transitions.values())
        normalized_reward = self.normalize_reward(reward, min_reward, max_reward)

        print("Normalised Reward = ", normalized_reward)

        return normalized_reward

    def normalize_reward(self, reward, min_reward, max_reward):
        # Avoid division by zero
        if min_reward == max_reward:
            return 0

        # Shift and scale the reward to the range [0, 1]
        scaled_reward = (reward - min_reward) / (max_reward - min_reward)

        # Apply the sigmoid function for smoother scaling
        sigmoid = lambda x: 1 / (1 + math.exp(-x))
        normalized_reward = 2 * sigmoid(scaled_reward) - 1

        return normalized_reward

    def q_learning(self, personality_vector, current_state, next_action_state):
        print("\nStarting with Q Learning ...")

        # Define current and next states
        print("\nPersonality Vector:", personality_vector)
        print("Current State:", current_state)
        print("Next Action State:", ActionStates.get_action(next_action_state))

        # Execute action and observe reward
        reward = self.calculate_reward(personality_vector, current_state[1].value, next_action_state)
        print("\nReward:", reward)

        # Update Q-value using Q-learning update rule
        # for dominant personality
        dominant_personality_index = PersonalityIndex.get_dominant_personality(personality_vector)
        q_range_index = Range.index_to_range_value(personality_vector[dominant_personality_index])
        print("\nDominant Personality :", PersonalityIndex.get_personality(dominant_personality_index))
        print("\nQ Range Index:", q_range_index)

        q_current = self.Q[dominant_personality_index][q_range_index][current_state[0].value][current_state[1].value][next_action_state]
        max_q_next = np.max(self.Q[dominant_personality_index][q_range_index][current_state[0].value][next_action_state])
        print("Q-value for current state-action pair:", q_current)
        print("Max Q-value for next state:", max_q_next)

        updated_q_value = (1 - self.alpha) * q_current + self.alpha * (reward + self.gamma * max_q_next)
        print("\nUpdated Q-value:", updated_q_value)

        self.Q[dominant_personality_index][q_range_index][current_state[0].value][current_state[1].value][next_action_state] = updated_q_value

        # Q table for 
        q_table_slice = self.Q[dominant_personality_index][q_range_index][current_state[0].value][current_state[1].value]
        print("Q table slice:")
        self.print_q_table(q_table_slice, current_state[0].value, current_state[1].value, dominant_personality_index)

        self.save_q_table()
