import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_TXT = BASE_DIR/"data"/"raw"/"samplepdf.txt"
OUTPUT_PASSAGES = BASE_DIR/"data"/"passages"/"samplepdf.passages.txt"

# number of chars per passage, about 150-200 tokens
TARGET_CHARS = 800

# takes in raw text and returns list of sentences
def simple_sentence_split(text: str):
    # split on .!? followed by space
    parts = re.split(r'([.?!])', text)
    sentences = []
    
    # every other element is a sentence, followed by a punctuation
    # Ex: {"Hello world", ".", "How are you", "?", ...}
    for i in range(0, len(parts)-1, 2):
        sent = (parts[i] + parts[i+1]).strip()
        if sent:
            sentences.append(sent)
    
    # checking if there's a trailing part without ending punctuation
    if len(parts) % 2 == 1 and parts[-1].strip():
        sentences.append(parts[-1].strip())
    return sentences

# grouping sentences into passages of about target_chars length, returns list of passages
def chunk_sentences(sentences, target_chars=TARGET_CHARS):
    
    # chunk holds sentences for current passage, length is current char count
    # and passages is final list of all passage strings
    chunk = []
    length = 0
    passages = []
    
    # checking if sentence can fit in current chunk
    for s in sentences:
        if length + len(s) > target_chars and chunk:
            passages.append(" ".join(chunk).strip())
            chunk = []
            length = 0
        # appending sentence to chunk and incrementing length
        chunk.append(s)
        length += len(s) + 1
    # appending last remaining chunk if it exists
    if chunk:
        passages.append(" ".join(chunk).strip())
    return passages

def main():
    OUTPUT_PASSAGES.parent.mkdir(parents=True, exist_ok=True)
    # read full text file into a string
    text = INPUT_TXT.read_text(encoding="utf-8")
    # split into rough sentences
    sents = simple_sentence_split(text)
    # create 800 char passages
    passages = chunk_sentences(sents, TARGET_CHARS)
    
    # write passages to output file, one passage per line
    with OUTPUT_PASSAGES.open("w", encoding="utf-8") as f:
        for p in passages:
            if p:
                f.write(p.replace("\n", " ").strip() + "\n")
    
    # print summary to console
    print(f"Write {len(passages)} passages -> {OUTPUT_PASSAGES}")

if __name__ == "__main__":
    main()