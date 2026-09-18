"""Generate a ~4 minute synthetic lecture/discussion for dry-testing the pipeline.

Uses macOS `say` with distinct voices per speaker, inserts silence gaps between
turns, and concatenates into one 16kHz mono wav.

Run: python3 gen_long_lecture.py
"""
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

OUT_DIR = Path(__file__).parent / "audio" / "lectures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR = Path("/tmp/long_lecture_chunks")
TMP_DIR.mkdir(exist_ok=True)

VOICES = {
    "Prof": "Daniel",
    "Priya": "Tara",
    "Arjun": "Rishi",
    "Rajan": "Fred",
}

# (speaker, text) — a longer, more detailed back-and-forth discussion.
SCRIPT = [
    ("Prof", "Today we will go deeper into the attention mechanism in transformers, and I want this to be a real discussion, not just me talking. Arjun, what is the core idea that attention is trying to capture?"),
    ("Arjun", "I think the core idea is that every token should be able to look at every other token directly, instead of only nearby ones like in a convolution or only the previous hidden state like in a recurrent network."),
    ("Prof", "Good. So this feels similar to convolutional networks, where locality is a built in bias. But here we are removing that bias entirely and letting the model attend to everything. Priya, how is the attention distribution actually computed?"),
    ("Priya", "So for each token we compute a query vector, and every token also has a key vector. We take the dot product of the query with every key, scale it down, and then run that through a softmax so it becomes a probability distribution over all the tokens."),
    ("Prof", "Exactly right. Now, are the query and key matrices learned weights, or are they fixed ahead of time?"),
    ("Rajan", "They have to be learned, otherwise the model could never adapt the notion of relevance to the actual task. If they were fixed, attention would just be some generic similarity measure that has nothing to do with what the model is trying to predict."),
    ("Prof", "Correct. They are learned. Priya, can you build on that and tell us what happens after we have the attention weights?"),
    ("Priya", "Once we have the softmax weights for a given token, we use them to take a weighted sum of the value vectors from all the other tokens. So the output for that token is really a blend of information pulled from everywhere else in the sequence, weighted by how relevant each position is."),
    ("Arjun", "Okay, got it. So intuitively the query is like asking a question, the key is like an index that says how well each position answers that question, and the value is the actual content that gets pulled in."),
    ("Prof", "That is a really good way to phrase it. Does anyone see a downside to letting every token attend to every other token like this?"),
    ("Rajan", "Doesn't that mean self attention has quadratic cost in the sequence length? Because every token is comparing itself against every other token, so the number of comparisons grows as the square of the sequence length."),
    ("Priya", "I agree, that seems expensive, especially for very long documents or for things like genomics data where the sequence can be tens of thousands of tokens long."),
    ("Prof", "Right, that quadratic cost is exactly the bottleneck that motivated a lot of later research. Arjun, have you come across any of the approaches people use to get around this?"),
    ("Arjun", "I have read a little about sparse attention, where instead of attending to everything, each token only attends to a fixed pattern of other tokens, like a local window plus a few global tokens. That brings the cost down closer to linear."),
    ("Prof", "Good, sparse attention is one direction. There is also linear attention, where people approximate the softmax with kernel tricks so the whole operation can be reordered to avoid the quadratic term. And more recently, state space models like Mamba try to replace attention entirely with a recurrent style mechanism that is linear in sequence length but still captures long range dependencies."),
    ("Rajan", "So is the takeaway that attention is powerful because it is global, but that same globality is also its biggest weakness at scale?"),
    ("Prof", "That is precisely the tension. It is a very good summary. Priya, before we move on, can you explain why we divide by the square root of the key dimension before the softmax?"),
    ("Priya", "If we don't scale it down, the dot products can get very large in magnitude when the dimension is high, and that pushes the softmax into a region where the gradients are extremely small, so the model becomes hard to train. Dividing by the square root of the dimension keeps the variance of the dot product roughly constant regardless of the dimension size."),
    ("Prof", "Exactly. That small detail turns out to matter a lot in practice. Okay, let's take this one step further. Arjun, what is the purpose of having multiple attention heads instead of just one?"),
    ("Arjun", "Each head can specialize in a different kind of relationship. One head might learn to track syntactic structure, like which word a pronoun refers to, while another head might track something more semantic, like topic similarity across the sentence. Using many heads in parallel lets the model capture several of these patterns at once instead of being forced into a single notion of relevance."),
    ("Prof", "Right, and empirically when people visualize attention heads in trained transformers, you do see this kind of specialization emerge, even though nothing in the architecture explicitly forces it. Rajan, any final thoughts before we wrap up this section?"),
    ("Rajan", "Just that it is interesting how such a simple operation, a weighted average based on similarity, ends up being the foundation for almost all of modern large language models. It feels almost too simple for how powerful it is."),
    ("Prof", "That is a great note to end on. Simplicity combined with scale is really the story of the last few years in this field. Thank you all, that was a genuinely good discussion."),
]


def main():
    chunks = []
    sr_target = 16000
    silence = np.zeros(int(1.6 * sr_target), dtype=np.float32)

    for i, (speaker, text) in enumerate(SCRIPT):
        voice = VOICES[speaker]
        aiff = TMP_DIR / f"{i:03d}.aiff"
        wav = TMP_DIR / f"{i:03d}.wav"
        subprocess.run(["say", "-v", voice, "-o", str(aiff), text], check=True)
        subprocess.run(
            ["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", str(aiff), str(wav)],
            check=True,
        )
        data, sr = sf.read(wav, dtype="float32")
        assert sr == sr_target
        chunks.append(data)
        chunks.append(silence)
        print(f"  [{i:02d}] {speaker:6s} ({voice}): {text[:60]}...")

    full = np.concatenate(chunks)
    out_path = OUT_DIR / "long_discussion.wav"
    sf.write(out_path, full, sr_target)
    duration = len(full) / sr_target
    print(f"\nWrote {out_path}  ({duration:.1f}s, {duration/60:.1f} min)")


if __name__ == "__main__":
    main()
