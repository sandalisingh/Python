#downloads mp3 in the directory

import os
from gtts import gTTS

Text = "crux is a program existing in my /home/ directory, and if I type the command crux tide-index into a terminal, it seems to work properly."

print("please wait...processing")
TTS = gTTS(text=Text, lang='en-uk')

# Save to mp3 in current dir.
TTS.save("voice.mp3")

# # Plays the mp3 using the default app on your system
# # that is linked to mp3s.
os.system("afplay voice.mp3")