from EmotionClassifier import EmotionClassifier
from States import logging, EmotionStates, Range
from typing import Tuple

class EmotionGenerator:
    def __init__(self, personality_vector: list[float], environment: str = "Resting"):
        self.PERSONALITY_VECTOR = personality_vector
        self.EMOTION_CLASSIFIER_MODEL = EmotionClassifier()
        self.EMOTION_TUPLE = (EmotionStates.Neutral, 0.0)  # (emotion, intensity)
        self.DECAY_RATE = 0.2
        self.MAX_INTENSITY = 1.0
        self.personality_modifier = self.get_personality_modifier()

        self.initialize_emotion(environment)

    def initialize_emotion(self, environment: str) -> None:
        emotional_perception = self.EMOTION_CLASSIFIER_MODEL.predict(environment)  
        emotional_perception_tuple = (emotional_perception, 0.8)
        # print("\nEmotional perception of env = ", emotional_perception_tuple)
        personality_modifier = self.personality_modifier
        self.EMOTION_TUPLE = self.weight_emotion(emotional_perception_tuple, personality_modifier)
        # print("\nInitial emotion tuple = ", self.EMOTION_TUPLE)

    def update_emotion(self, chat_text: str) -> None:
        emotional_perception = self.EMOTION_CLASSIFIER_MODEL.predict(chat_text)
        emotional_perception_tuple = (emotional_perception, 0.5)
        # print("\nEmotional perception of people = ", emotional_perception_tuple)
        # personality_modifier = self.personality_modifier
        weighted_emotional_perception = self.weight_emotion(emotional_perception_tuple)
        self.EMOTION_TUPLE = self.decay_emotion(self.EMOTION_TUPLE)
        self.EMOTION_TUPLE = self.combine_emotions(self.EMOTION_TUPLE, weighted_emotional_perception)
        # print("\nUpdated emotion tuple = ", self.EMOTION_TUPLE)

    def get_personality_modifier(self) -> float:
        openness_range = self.PERSONALITY_VECTOR[0]
        conscientiousness_range = self.PERSONALITY_VECTOR[1]
        extraversion_range = self.PERSONALITY_VECTOR[2]
        agreeableness_range = self.PERSONALITY_VECTOR[3]
        neuroticism_range = self.PERSONALITY_VECTOR[4]

        # Define modifiers based on personality ranges
        openness_modifier = 0.0  # Less susceptible to environment when open
        conscientiousness_modifier = 0.1  # More influenced by negative emotions
        extraversion_modifier = 0.2  # More influenced by others' emotions
        agreeableness_modifier = 0.1  # More easily swayed by positive emotions
        neuroticism_modifier = -0.1  # Prone to negative emotions, but modifier counters it slightly

        # Adjust modifiers based on ranges (consider using weights)
        if openness_range in Range.High.value:
            openness_modifier -= 0.2
        elif openness_range in Range.Low.value:
            openness_modifier += 0.2

        if conscientiousness_range in Range.High.value:
            conscientiousness_modifier = min(conscientiousness_modifier, 0.3)  # Cap for high conscientiousness
        elif conscientiousness_range in Range.Low.value:
            conscientiousness_modifier = max(conscientiousness_modifier, -0.1)  # Floor for low conscientiousness

        if extraversion_range in Range.High.value:
            extraversion_modifier = min(extraversion_modifier, 0.4)  # Cap for high extraversion
        elif extraversion_range in Range.Low.value:
            extraversion_modifier = max(extraversion_modifier, 0.0)  # Floor for low extraversion

        if agreeableness_range in Range.High.value:
            agreeableness_modifier = min(agreeableness_modifier, 0.2)  # Cap for high agreeableness
        elif agreeableness_range in Range.Low.value:
            agreeableness_modifier = max(agreeableness_modifier, -0.2)  # Floor for low agreeableness

        if neuroticism_range in Range.High.value:
            neuroticism_modifier = min(neuroticism_modifier, -0.3)  # Stronger modifier for high neuroticism
        elif neuroticism_range in Range.Low.value:
            neuroticism_modifier = max(neuroticism_modifier, 0.0)  # No negative modifier for low neuroticism

        # Combine modifiers using a weighted average
        total_modifier = sum([openness_modifier, conscientiousness_modifier, extraversion_modifier, agreeableness_modifier, neuroticism_modifier])
        total_modifier /= 5.0

        # print("\nPersonality modifier = ", total_modifier)

        return total_modifier

    def combine_emotions(self, emotion_tuple_1: Tuple[EmotionStates, float], emotion_tuple_2: Tuple[EmotionStates, float], weight1: float = 0.5, weight2: float = 0.5) -> Tuple[EmotionStates, float]:
        combined_emotion = emotion_tuple_1[0]  # Initialize with emotion from emotion1
        intensity1, intensity2 = emotion_tuple_1[1], emotion_tuple_2[1]
        # print("\nEmotion tuple 1 = ", emotion_tuple_1)
        # print("Emotion tuple 2 = ", emotion_tuple_2)

        # Weighted average of intensities, clamped between 0 and MAX_INTENSITY
        combined_intensity = (weight1 * intensity1 + weight2 * intensity2)
        combined_intensity = max(0.0, min(combined_intensity, self.MAX_INTENSITY))

        # print("Combined intensity = ", combined_intensity)

        # If resulting intensity is high enough, choose the more intense emotion
        if intensity2 > intensity1:
            combined_emotion = emotion_tuple_2[0]
            # print("Combined emotion = ", combined_emotion)

        combined_emotion_tuple = (combined_emotion, combined_intensity)

        # print("Combined emotion tuple = ", combined_emotion_tuple)

        return combined_emotion_tuple

    def weight_emotion(self, emotion_tuple: Tuple[EmotionStates, float]) -> Tuple[EmotionStates, float]:
        emotion_type = emotion_tuple[0]
        intensity = emotion_tuple[1]

        # Apply weight based on openness (more influenced by external emotions with higher openness)
        openness_weight = self.MAX_INTENSITY - abs(self.PERSONALITY_VECTOR[0]/10 - 0.5)

        # Apply weight based on agreeableness (more influenced by positive emotions with higher agreeableness)
        agreeableness_weight = self.PERSONALITY_VECTOR[3]

        # Apply weight based on neuroticism (more susceptible to emotional fluctuations with higher neuroticism)
        neuroticism_weight = abs(self.PERSONALITY_VECTOR[4])

        # Combine weights and clamp between 0 and 1
        combined_weight = max(0.0, min(openness_weight + agreeableness_weight + neuroticism_weight, self.MAX_INTENSITY))

        # Apply weight to intensity and personality modifier
        weighted_intensity = intensity + (combined_weight * self.personality_modifier)

        # Clamp weighted intensity between 0 and MAX_INTENSITY
        weighted_intensity = max(0.0, min(weighted_intensity, self.MAX_INTENSITY))

        weighted_emotion_tuple = (emotion_type, weighted_intensity)

        # print("Weighted emotion tuple = ", weighted_emotion_tuple)

        return weighted_emotion_tuple

    def decay_emotion(self, emotion_tuple: Tuple[EmotionStates, float]) -> Tuple[EmotionStates, float]:
        emotion, intensity = emotion_tuple

        # Reduce intensity by DECAY_RATE
        decayed_intensity = max(0.0, intensity - self.DECAY_RATE)

        decayed_emotion_tuple = (emotion, decayed_intensity)

        # print("\nDecayed emotion tuple = ", decayed_emotion_tuple)

        return decayed_emotion_tuple

    def get_current_emotion_as_string(self):
        return EmotionStates.index_to_enum(self.EMOTION_TUPLE[0].value)

    def get_current_emotion_emoji(self):
        return EmotionStates.get_emoji(self.EMOTION_TUPLE[0])
