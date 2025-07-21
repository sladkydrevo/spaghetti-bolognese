# Ragu' alla Bolognese

An attempt to create something useful and delicious... BEEF RAGU'! RAG, in short. 
We gotta evaluate these Italian restaurants, they can't get away with this.


notes.

### QQ/A dataset
Expected answers are written in the most explicit and simple way as possible, which means short sentences using the keywords.
That insures the lexical comparison accuracy.

Each comparison metric represents the distance of two compared answers from 0 to 1, with 0 meaning the texts are completely different and 1 meaning the texts are identical in that metric.
Dice coefficient is used to measure the lexical similarity in texts. It is sensitive to vocabulary overlap, which makes it the best choice for the comparison.
