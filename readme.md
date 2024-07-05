# Argument Mining Survey


## Argument Mining Papers
- [Argument mining using BERT and self-attention based embeddings](https://arxiv.org/pdf/2302.13906)
- [Can Large Language Models perform Relation-based Argument Mining?](https://arxiv.org/pdf/2402.11243)
- [In-Context Learning and Fine-Tuning GPT for Argument Mining](https://arxiv.org/abs/2406.06699)
- [Argument mining: A survey](https://direct.mit.edu/coli/article-pdf/45/4/765/1847520/coli_a_00364.pdf)
## Distilling LLM
- [D2LLM: Decomposed and Distilled Large Language Models for Semantic Search](https://arxiv.org/pdf/2406.17262)

## Scoring Mechanism
- [Language models and automated essay scoring](https://arxiv.org/pdf/1909.09482)
### Introduction Stage
1. Clarity
- **Vocabulary and Language Use**: Assess the debater's choice of words and language use. Look for a varied vocabulary, appropriate use of terminology, and avoidance of overly complex or obscure language.
    
    - Lexical Richness Measures  
    ``` 
    the total number of words in the text (w) and the number of unique terms (t).
    TTR = t/w
    Herdan= log(t) / log(w)
    Dugast = log(w) * log(w) / log(w) - log(t) 
    ```
    - word complexity
    ```
    average word length = sum(word_lengths) / len(words)
    ```
    - use of rare
    ```
    complex_words = [word for word in words_in_text if word.lower() not in word_list]
    ```
    - Readability Score (Flesch-Kincaid Grade Level)
    ```
    total words = w, syllables = s, total sentences = t
    FKGL = 0.39 * w / t + 11.8 * s / w - 15.59
    ```

- **Coherence of Arguments**: Evaluate how well the debater organizes and structures their arguments. Check for logical flow, transitions between points, and a clear progression of ideas.
    - BERT-based Coherence Models: Use pre-trained BERT models fine-tuned for coherence scoring to evaluate the logical flow between sentences and paragraphs.
    - Node and Edge Representation: Represent each argument as a node and the logical connections between them as edges. A well-structured argument will form a connected graph with minimal cycles.
    Centrality Measures: Use centrality measures (e.g., degree centrality, betweenness centrality) to identify key arguments and their influence on the overall structure.

- **Elimination of Jargon**: Consider whether the debater avoids using specialized terminology or jargon that may confuse the audience. Clear and concise language accessible to a wider audience is key.
- **Clarity of Explanation**: Evaluate how effectively the debater explains their points and ideas. Look for clear, straightforward explanations that help the audience understand complex topics.
2. Conciseness
- **Avoidance of Redundancy**: Assess whether the debater repeats information unnecessarily or uses redundant phrases, as this can impact the conciseness of their communication.
- **Focus on Key Points**: Evaluate the debater's ability to prioritize key arguments and avoid getting sidetracked by tangential or irrelevant information.
- **Clarity of Expression**: Consider how clearly the debater communicates their ideas while maintaining brevity. Confusing or convoluted explanations may indicate a lack of conciseness.
- **Logical Flow**: Evaluate if the debater's speech follows a logical progression, presenting arguments in a structured and organized manner that enhances conciseness.
3. Logical Coherence
- **Flow of Argument**: Look for a clear structure in their arguments
- **Consistency**: Check for consistency in their reasoning throughout the debate. 
- **Clarity and Precision**: Assess the clarity and precision of their language.
- **Logical Fallacies**: Be mindful of logical fallacies such as ad hominem attacks, slippery slope arguments, straw man fallacies, or appeals to emotion. If the speaker relies on such fallacious reasoning, it can indicate a lack of logical coherence.
### Rebuttal
1. Effectiveness
2. Evidence and Support
3. Confidence
### Conclusion
1. Logical Coherence
2. Conciseness
3. Impactfulness




## Project
- [Sequence-to-Sequence Generative Dialogue Systems](https://github.com/vliu15/dialogue-seq2seq)
- [Extraction and use of arguments in Recommender Systems](https://github.com/argrecsys/arg-miner)
- [ArgMiner: End-to-end Argument Mining](https://github.com/namiyousef/argument-mining)