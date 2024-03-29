from enum import Enum
import numpy as np
from termcolor import colored  

def logging(level, message):
    if level == 'info':
        print(colored("\nINFO : "+message+"\n", "yellow"))  # Logging info in yellow
    elif level == 'error':
        print(colored("\nERROR : "+message+"\n", "red"))     # Logging error in red

class ActionStates(Enum):
    Patrolling = 0
    Attacking = 1
    Fleeing = 2
    Alerted = 3
    Searching = 4
    Interacting = 5
    Resting = 6
    Following = 7
    Celebrating = 8
    Helping = 9

    @staticmethod
    def get_action(index):
        for action in ActionStates:
            if action.value == index:
                return action.name
        return None

    @staticmethod
    def get_action_index(action_name):
        try:
            action_name = str(action_name).strip().capitalize()
            return ActionStates[action_name].value
        except Exception as e:
            logging("error", str(e))
            return None

class PersonalityIndex(Enum):
    Openness = 0
    Conscientiousness = 1
    Extraversion = 2
    Agreeableness = 3
    Neuroticism = 4

    @staticmethod
    def get_personality(index):
        for p in PersonalityIndex:
            if p.value == index:
                return p.name
        return None

    @staticmethod
    def get_dominant_personality(personality_vector):
        dominant_personality_index = np.argmax(personality_vector)
        return dominant_personality_index

class Range(Enum):
    Low = range(0, 4)
    Medium = range(4, 8)
    High = range(8, 10)

    @staticmethod
    def index_to_range_value(index):
        if index in Range.Low.value:
            return 0
        elif index in Range.Medium.value:
            return 1
        elif index in Range.High.value:
            return 2

    @staticmethod
    def index_to_range_key(index):
        if index in Range.Low.value:
            return Range.Low
        elif index in Range.Medium.value:
            return Range.Medium
        elif index in Range.High.value:
            return Range.High

class EmotionStates(Enum):
    Admiration = 0
    Amusement = 1
    Anger = 2
    Annoyance = 3
    Approval = 4
    Caring = 5
    Confusion = 6
    Curiosity = 7
    Desire = 8
    Disappointment = 9
    Disapproval = 10
    Disgust = 11
    Embarrassment = 12
    Excitement = 13
    Fear = 14
    Gratitude = 15
    Grief = 16
    Joy = 17
    Love = 18
    Nervousness = 19
    Optimism = 20
    Pride = 21
    Realization = 22
    Relief = 23
    Remorse = 24
    Sadness = 25
    Surprise = 26
    Neutral = 27

    @staticmethod
    def get_emotion(index):
        for emotion in EmotionStates:
            if emotion.value == index:
                return emotion.name
        return None

    @staticmethod
    def get_emotion_index(emotion_name):
        try:
            emotion_name = str(emotion_name).strip().capitalize()
            return EmotionStates[emotion_name].value
        except Exception as e:
            logging("error", str(e))
            return None

    @staticmethod
    def string_to_enum(emotion_str):
        try:
            key = str(emotion_str).strip().capitalize()
            return EmotionStates[key]
        except Exception as e:
            logging("error", str(e))
            return None

    @staticmethod
    def get_emoji(emotion):
        emojis = {
            EmotionStates.Admiration: '😊',
            EmotionStates.Amusement: '😄',
            EmotionStates.Anger: '😠',
            EmotionStates.Annoyance: '😒',
            EmotionStates.Approval: '👍',
            EmotionStates.Caring: '❤️',
            EmotionStates.Confusion: '😕',
            EmotionStates.Curiosity: '🤔',
            EmotionStates.Desire: '😏',
            EmotionStates.Disappointment: '😞',
            EmotionStates.Disapproval: '👎',
            EmotionStates.Disgust: '🤢',
            EmotionStates.Embarrassment: '😳',
            EmotionStates.Excitement: '😃',
            EmotionStates.Fear: '😨',
            EmotionStates.Gratitude: '🙏',
            EmotionStates.Grief: '😢',
            EmotionStates.Joy: '😂',
            EmotionStates.Love: '😍',
            EmotionStates.Nervousness: '😰',
            EmotionStates.Optimism: '😊',
            EmotionStates.Pride: '🦁',
            EmotionStates.Realization: '😮',
            EmotionStates.Relief: '😅',
            EmotionStates.Remorse: '😔',
            EmotionStates.Sadness: '😢',
            EmotionStates.Surprise: '😲',
            EmotionStates.Neutral: '😐',
        }
        return emojis.get(emotion, '')

    @staticmethod
    def get_perception(emotion):
        perceptions = {
            EmotionStates.Admiration: "I admire.",
            EmotionStates.Amusement: "I find it amusing.",
            EmotionStates.Anger: "I'm angry.",
            EmotionStates.Annoyance: "I'm annoyed.",
            EmotionStates.Approval: "I approve.",
            EmotionStates.Caring: "I care deeply.",
            EmotionStates.Confusion: "I'm confused.",
            EmotionStates.Curiosity: "I'm curious.",
            EmotionStates.Desire: "I desire.",
            EmotionStates.Disappointment: "I'm disappointed.",
            EmotionStates.Disapproval: "I disapprove.",
            EmotionStates.Disgust: "I'm disgusted.",
            EmotionStates.Embarrassment: "I'm embarrassed.",
            EmotionStates.Excitement: "I'm excited.",
            EmotionStates.Fear: "I'm afraid.",
            EmotionStates.Gratitude: "I'm grateful.",
            EmotionStates.Grief: "I'm grieving.",
            EmotionStates.Joy: "I'm filled with joy.",
            EmotionStates.Love: "I'm in love.",
            EmotionStates.Nervousness: "I'm nervous.",
            EmotionStates.Optimism: "I'm optimistic.",
            EmotionStates.Pride: "I'm proud.",
            EmotionStates.Realization: "I've realized.",
            EmotionStates.Relief: "I feel relieved.",
            EmotionStates.Remorse: "I feel remorseful.",
            EmotionStates.Sadness: "I'm sad.",
            EmotionStates.Surprise: "I'm surprised.",
            EmotionStates.Neutral: "I feel neutral.",
        }
        return perceptions.get(emotion, '')
