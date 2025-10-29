from transformers import pipeline
import torch, scipy
import soundfile as sf

pipe = pipeline(
    "text-to-audio",
    model="facebook/musicgen-large",
    # device="cuda",
    model_kwargs={
        "load_in_4bit": True,                 # <- 使用 bitsandbytes 4-bit
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_compute_dtype": torch.float16,
    }
)

music = pipe("The instrument heard is a flute with a melody described as simple and poignant. The music's genre includes easy listening, electronic, jazz, lounge, and world music, with a mood fitting for film. It has a time signature of 4/4 and a tempo of approximately 166.7 BPM. The key of the piece is E minor, and it features chords such as E major, G major, D major, C major, and B minor. The form of the piece is in A minor, with a recurring progression of D major and E major chords. There's also a notable shift to F# major. The flute's timbre is breathy and reverberant, indicating it may have been recorded in a large room or space. Additionally, there's a noticeable dynamic range from soft to loud articulations.", forward_params={"do_sample": True})
# scipy.io.wavfile.write("musicgen_out.wav", rate=music["sampling_rate"], data=music["audio"])
print(music['audio'].shape)
music['audio'] = music['audio'].squeeze()
sf.write('musicgensf.wav', music['audio'], samplerate=music['sampling_rate'])
