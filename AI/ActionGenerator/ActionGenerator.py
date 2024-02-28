import numpy as np
from States import ActionStates, Range, PersonalityIndex, index_to_range_value, index_to_range_key, get_action, get_personality

class ActionGenerator:

    def __init__(self, load_from_file=True):
        # Initialize Q-table
        self.no_of_personality_states = 5    # OCEAN personality model
        self.no_of_ranges_of_personality_states = 3  # 3 ranges for each personality states (0-3, 4-7, 8-10)
        self.no_of_emotional_states = 28  # 10 emotional states
        self.no_of_action_states = 10  # 10 action states

        if load_from_file:
            try:
                self.Q = np.load('Action_Q_Table.npy')
                print("\nQ-table loaded successfully.\n")
            except FileNotFoundError:
                print("\nQ-table not found. Initializing new Q-table.")
                self.Q = self.initialize_q_table()
        else:
            self.Q = self.initialize_q_table()

        # Parameters
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.9  # Discount factor
        self.epsilon = 0.1  # Exploration rate

    def initialize_q_table(self):
        # initialize Q table
        # 5 ocean personality * 3 ranges (0-3,4-7,8-10) * 10 emotional states * 10 FROM action states * 10 TO action states
        return np.random.random((self.no_of_personality_states, self.no_of_ranges_of_personality_states,
                                 self.no_of_emotional_states, self.no_of_action_states, self.no_of_action_states))

    def save_q_table(self):
        np.save('Action_Q_Table.npy', self.Q)
        print("Q-table saved successfully.\n")
    
    def calculate_reward(self, personality_vector, current_action, next_action):
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
            # Add more transitions as needed
        }

        # Check if the current and next actions correspond to a defined transition
        transition_key = (current_action, next_action)
        if transition_key in transitions:
            reward = transitions[transition_key]

        print("\nTransition Reward = ", reward)

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
            # print("Personality_vector["+str(trait)+"] = "+str(personality_vector[trait]))
            index = index_to_range_key(personality_vector[trait])
            # print("Mapping - trait:"+str(trait)+" to index:"+str(index))
            if next_action in preferences[index]:
                reward += 1

        print("Preference Reward = ", reward)
        
        # Normalize the total reward to the range [-1, 1]
        max_reward = max(reward for reward in transitions.values()) + len(personality_preferences)  # Add maximum additional reward
        min_reward = min(reward for reward in transitions.values())
        if max_reward != min_reward:
            normalized_reward = 2 * (reward - min_reward) / (max_reward - min_reward) - 1
        else:
            normalized_reward = 0  # Handle the case where all rewards are the same

        print("Normalised Reward = ", reward)
        print("\n")

        return normalized_reward

    def action_generator(self, personality_vector, emotional_state_index, previous_action_state_index) :
        # Choose action based on epsilon-greedy policy
        if np.random.rand() < self.epsilon: 
            final_action_state_index = np.random.randint(self.no_of_action_states)  # Choose random action
            print("\nRandom final action state index = ", final_action_state_index)
        else:
            # initialize an array for 10 action states with value 1
            q_val_array = np.ones(10)

            for i in (0,4):     # 5 ocean personalities
                for j in (0,9) :    # 10 action states
                    # Q value of jTH action state = product of Q value of the jTH action state of each ocean personality
                    
                    personality_index = index_to_range_value(personality_vector[i])

                    # print("i - ", i)
                    # print("Personality index - ", personality_index)
                    # print("Emotional state index - ", emotional_state_index)
                    # print("Prev axn state index - ", previous_action_state_index)
                    # print("j - ", j)
                    q_val_array[j] *= self.Q[i][personality_index][emotional_state_index.value][previous_action_state_index.value][j]
            
            # select action state with max q value
            final_action_state_index = np.argmax(q_val_array)  # gives the index of that action state
            print("\nFinal action state index = ", final_action_state_index)

        print("\n")
        return final_action_state_index

    def get_dominant_personality(self, personality_vector) :
        dominant_personality_index = np.argmax(personality_vector)
        print("\nDominant Personality = ", dominant_personality_index)
        print("\n")
        return dominant_personality_index

    def q_learning(self, personality_vector, current_state, next_action_state):
        print("\nStarting with Q Learning ...")

        # Define current and next states
        print("\nPersonality Vector:", personality_vector)
        print("Current State:", current_state)
        print("Next Action State:", get_action(next_action_state))

        # Execute action and observe reward
        reward = self.calculate_reward(personality_vector, current_state[1].value, next_action_state)
        print("\nReward:", reward)

        # Update Q-value using Q-learning update rule
        # for dominant personality
        dominant_personality_index = self.get_dominant_personality(personality_vector)
        q_range_index = index_to_range_value(personality_vector[dominant_personality_index])
        print("\nDominant Personality :", get_personality(dominant_personality_index))
        print("\nQ Range Index:", q_range_index)

        q_current = self.Q[dominant_personality_index][q_range_index][current_state[0].value][current_state[1].value][next_action_state]
        max_q_next = np.max(self.Q[dominant_personality_index][q_range_index][current_state[0].value][next_action_state])
        print("Q-value for current state-action pair:", q_current)
        print("Max Q-value for next state:", max_q_next)

        updated_q_value = (1 - self.alpha) * q_current + self.alpha * (reward + self.gamma * max_q_next)
        print("\nUpdated Q-value:", updated_q_value)

        self.Q[dominant_personality_index][q_range_index][current_state[0].value][current_state[1].value][next_action_state] = updated_q_value

        # Q table for 
        q_table_slice = self.Q[dominant_personality_index][q_range_index][current_state[0].value]
        # print("Q table slice:")
        # print(q_table_slice)

        self.save_q_table()
