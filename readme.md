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
- [Exploring Relevance and Coherence for Automated Text Scoring using Multi-task Learning](https://ksiresearch.org/seke/seke22paper/paper024.pdf)
- [Transformer Models for Text Coherence Assessment](https://arxiv.org/pdf/2109.02176)
- [Improving Unsupervised Dialogue Topic Segmentation with Utterance-Pair Coherence Scoring](https://arxiv.org/pdf/2106.06719)
- [Automatic Evaluation of Topic Coherence](https://aclanthology.org/N10-1012.pdf)
- [Coh-Metrix: Automated cohesion and coherence scores to predict text readability and facilitate comprehension](https://www.academia.edu/download/30813764/IESproposal.pdf)
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

    - N-gram Analysis: Identify repeated n-grams (sequences of n words) within the text. Higher frequency of repeated n-grams indicates redundancy.
    - Cosine Similarity: Calculate the cosine similarity between different parts of the text. High similarity between different sections can indicate redundancy.
    - Jaccard Similarity: Measure the similarity between sets of words in different sentences or paragraphs. High Jaccard similarity can indicate redundancy.
    ``` 
    Example Calculation
    Preprocessing:

    from sklearn.feature_extraction.text import TfidfVectorizer
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    import string

    text = "Your text here."
    sentences = sent_tokenize(text)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    def preprocess(text):
        words = word_tokenize(text)
        words = [lemmatizer.lemmatize(word.lower()) for word in words if word.lower() not in stop_words and word not in string.punctuation]
        return ' '.join(words)

    preprocessed_sentences = [preprocess(sentence) for sentence in sentences]
    Vectorization:

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(preprocessed_sentences)
    Redundancy Detection:

    Cosine Similarity:

    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    cosine_similarities = cosine_similarity(tfidf_matrix)
    np.fill_diagonal(cosine_similarities, 0)  # Ignore self-similarity
    redundancy_score = np.mean(cosine_similarities)
    Jaccard Similarity:

    def jaccard_similarity(set1, set2):
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union

    jaccard_scores = []
    for i in range(len(preprocessed_sentences)):
        for j in range(i + 1, len(preprocessed_sentences)):
            set1 = set(preprocessed_sentences[i].split())
            set2 = set(preprocessed_sentences[j].split())
            jaccard_scores.append(jaccard_similarity(set1, set2))

    redundancy_score = np.mean(jaccard_scores)
    Scoring:

    Define a threshold (e.g., 0.5) for what constitutes high redundancy.
    The final redundancy score can be normalized or scaled as needed.
    Final Score Interpretation
    Low Redundancy Score: Indicates good avoidance of redundancy.
    High Redundancy Score: Indicates poor avoidance of redundancy, with repeated information.
    ```

    - **Focus on Key Points**: Evaluate the debater's ability to prioritize key arguments and avoid getting sidetracked by tangential or irrelevant information.
        - Relevance Scoring:

            Calculate cosine similarity or use BERT embeddings to score each segment against the key points.
        - Tangential Information Detection:

            Identify segments that do not align with the key points.
        - Scoring:

            Relevance Score: Sum of relevance scores for each segment.
            Tangential Score: Number of tangential segments.
            Final Score: Combine these scores, e.g., Final Score = Relevance Score - Tangential Score.
            . Preprocessing
        1. Text Cleaning: Remove any irrelevant information such as special characters, stop words, and perform tokenization.
        Sentence Segmentation: Split the text into sentences to analyze each statement individually.
        2. Segment Extraction
        Topic Modeling: Use models like Latent Dirichlet Allocation (LDA) to identify different topics discussed in the debate.
        Text Segmentation: Apply algorithms like TextTiling or BERT-based models to segment the text into coherent sections.
        ```
            Steps to perform Topic Modeling using LDA:

        Preprocess the Text:

        Tokenize the text.
        Remove stop words, punctuation, and other non-essential elements.
        Perform stemming or lemmatization.
        Convert Text to Document-Term Matrix:

        Use tools like CountVectorizer or TfidfVectorizer from libraries such as scikit-learn.
        Apply LDA:

        Use the LatentDirichletAllocation class from the sklearn.decomposition module.
        Fit the LDA model to the document-term matrix.
        Extract the topics and the associated words.
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.decomposition import LatentDirichletAllocation

        # Sample text data
        documents = ["Text of the debate goes here..."]

        # Preprocess and convert to document-term matrix
        vectorizer = CountVectorizer(stop_words='english')
        doc_term_matrix = vectorizer.fit_transform(documents)

        # Apply LDA
        lda = LatentDirichletAllocation(n_components=5, random_state=42)
        lda.fit(doc_term_matrix)

        # Display topics
        for idx, topic in enumerate(lda.components_):
            print(f"Topic {idx}:")
            print([vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-10:]])
        Text Segmentation
        To segment the text into coherent sections, you can apply algorithms like TextTiling or BERT-based models. TextTiling is a domain-independent algorithm for partitioning expository texts into multi-paragraph segments that represent subtopics.

        Steps to perform Text Segmentation using TextTiling:

        Preprocess the Text:

        Tokenize the text into sentences and paragraphs.
        Apply TextTiling:

        Use the nltk library which provides an implementation of TextTiling.
        import nltk
        from nltk.tokenize import TextTilingTokenizer

        # Sample text data
        text = "Text of the debate goes here..."

        # Apply TextTiling
        tt = TextTilingTokenizer()
        segments = tt.tokenize(text)

        # Display segments
        for i, segment in enumerate(segments):
            print(f"Segment {i}:")
            print(segment)
        ```

        3. Key Point Extraction
        Named Entity Recognition (NER): Identify and extract entities such as names, dates, and locations which are often key points in debates.
        Keyword Extraction: Use techniques like TF-IDF, RAKE, or BERT embeddings to extract important keywords from each segment.
        Summarization: Apply extractive or abstractive summarization models to generate concise summaries of each segment. Models like BERTSUM or T5 can be useful here.
        4. Sentiment Analysis
        Sentiment Analysis: Analyze the sentiment of each segment to understand the stance or emotion behind the statements. Models like VADER or BERT-based sentiment classifiers can be used.
        5. Argument Mining
        Argument Detection: Use argument mining techniques to identify claims, premises, and conclusions within the text. Models like Argumentative Zoning or BERT-based classifiers can help in this task.
        ```
        Example Code (Python with NLP Libraries)
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        # Example segments and key points
        segments = [
            "In this debate, we need to focus on climate change.",
            "The impact on polar bears is significant.",
            "However, let's also consider the economic implications.",
            "Speaking of the economy, the stock market has been volatile."
        ]
        key_points = ["climate change", "impact on polar bears", "economic implications"]

        # Vectorize the text
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(segments + key_points)

        # Calculate cosine similarity
        similarity_matrix = cosine_similarity(X[:len(segments)], X[len(segments):])

        # Calculate relevance scores
        relevance_scores = similarity_matrix.max(axis=1)

        # Detect tangential information (example threshold)
        tangential_threshold = 0.2
        tangential_segments = (relevance_scores < tangential_threshold).sum()

        # Final score
        final_score = relevance_scores.sum() - tangential_segments

        print(f"Relevance Scores: {relevance_scores}")
        print(f"Tangential Segments: {tangential_segments}")
        print(f"Final Score: {final_score}")
        ```
    - **Clarity of Expression**: Consider how clearly the debater communicates their ideas while maintaining brevity. Confusing or convoluted explanations may indicate a lack of conciseness.
    - **Logical Flow**: Evaluate if the debater's speech follows a logical progression, presenting arguments in a structured and organized manner that enhances conciseness.
3. Logical Coherence
    - **Flow of Argument**: Look for a clear structure in their arguments
    - **Consistency**: Check for consistency in their reasoning throughout the debate. 
    - **Clarity and Precision**: Assess the clarity and precision of their language.
    - **Logical Fallacies**: Be mindful of logical fallacies such as ad hominem attacks, slippery slope arguments, straw man fallacies, or appeals to emotion. If the speaker relies on such fallacious reasoning, it can indicate a lack of logical coherence.
### Rebuttal
1. Effectiveness
 - Respect and Professionalism:

    Is the rebuttal delivered respectfully without personal attacks?
    Does the speaker maintain a professional demeanor throughout?
 - Clarity and Coherence

    Method: Use Text Summarization and Readability Scores.
    Tools:
    Text Summarization: BERT, GPT-3, or other transformer-based models.
    Readability Scores: Flesch-Kincaid, Gunning Fog Index.
    Implementation:
    ```
    from transformers import pipeline
    summarizer = pipeline("summarization")
    summary = summarizer(text, max_length=50, min_length=25, do_sample=False)
    ```
 - Relevance to Opponent's Argument

    Method: Semantic Similarity and Topic Modeling.
    Tools:
    Semantic Similarity: BERT embeddings, cosine similarity.
    Topic Modeling: LDA (Latent Dirichlet Allocation).
    Implementation:
    ```
    from sentence_transformers import SentenceTransformer, util
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    embeddings1 = model.encode(text1)
    embeddings2 = model.encode(text2)
    similarity = util.pytorch_cos_sim(embeddings1, embeddings2)
    ```
 - Use of Evidence

    Method: Fact-Checking and Named Entity Recognition (NER).
    Tools:
    Fact-Checking: Fact-checking APIs like FactMata.
    NER: SpaCy, NLTK.
    Implementation:
    ```
    import spacy
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    for ent in doc.ents:
        print(ent.text, ent.label_)
    ```
 - Emotional Appeal

    Method: Sentiment Analysis and Emotion Detection.
    Tools:
    Sentiment Analysis: VADER, TextBlob.
    Emotion Detection: NRC Emotion Lexicon, transformer-based models.
    Implementation:
    ```
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()
    sentiment = analyzer.polarity_scores(text)
    ```
 -  Language and Tone

    Method: Tone Analysis and Politeness Detection.
    Tools:
    Tone Analysis: IBM Watson Tone Analyzer.
    Politeness Detection: Politeness API, custom classifiers.
    Implementation:
    ```
    from ibm_watson import ToneAnalyzerV3
    tone_analyzer = ToneAnalyzerV3(
        version='2017-09-21',
        iam_apikey='your_api_key',
        url='https://gateway.watsonplatform.net/tone-analyzer/api'
    )
    tone_analysis = tone_analyzer.tone(
        {'text': text},
        content_type='application/json'
    ).get_result()
    ```
 -   Engagement and Responsiveness

    Method: Turn-Taking Analysis and Interruption Detection.
    Tools:
    Turn-Taking Analysis: Custom scripts to analyze dialogue structure.
    Interruption Detection: Speech-to-Text APIs, custom classifiers.
2. Evidence and Support
 - Preprocessing the Speech Text
Tokenization: Split the speech into individual words or tokens.
Stop Words Removal: Remove common words that do not contribute to the meaning (e.g., "and", "the").
Stemming/Lemmatization: Reduce words to their base or root form.
 - Identifying Evidence and Support
Named Entity Recognition (NER): Use NER to identify entities such as dates, names, places, and organizations which often indicate evidence.
Keyword Matching: Look for keywords and phrases that typically indicate evidence and support, such as "according to", "research shows", "studies indicate", etc.
Citation Detection: Detect citations or references to external sources.
 - Scoring Mechanism
Frequency Count: Count the number of times evidence-related keywords and entities appear.
Sentiment Analysis: Analyze the sentiment around the evidence to ensure it is used positively and constructively.
Contextual Analysis: Use context to ensure that the identified evidence is relevant and supports the argument.
 - Mathematical Formulas
Evidence Score (ES): Calculate a score based on the frequency and relevance of evidence. [ ES = \sum_{i=1}^{n} \left( w_i \times f_i \right) ] where ( w_i ) is the weight assigned to the type of evidence (e.g., peer-reviewed study, expert opinion) and ( f_i ) is the frequency of that evidence type.

Support Score (SS): Evaluate how well the evidence supports the argument. [ SS = \frac{\text{Number of supported claims}}{\text{Total number of claims}} ]

 - Combining Scores
Overall Evidence and Support Score (OESS): Combine the Evidence Score and Support Score to get an overall score. [ OESS = \alpha \times ES + \beta \times SS ] where ( \alpha ) and ( \beta ) are weights that can be adjusted based on the importance of evidence and support in the context of the debate.
Example Implementation
```
import spacy
from collections import Counter

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Sample speech text
speech_text = "According to a study by Harvard, climate change is accelerating. Experts agree that immediate action is necessary."

# Preprocess text
doc = nlp(speech_text)
tokens = [token.lemma_ for token in doc if not token.is_stop]

# Identify evidence and support
evidence_keywords = ["according to", "study", "research", "experts", "data"]
evidence_entities = [ent.text for ent in doc.ents if ent.label_ in ["ORG", "PERSON", "DATE"]]

# Calculate Evidence Score
evidence_count = Counter(tokens + evidence_entities)
evidence_score = sum(evidence_count[key] for key in evidence_keywords)

# Calculate Support Score (simplified example)
total_claims = 2  # Example number of claims
supported_claims = 1  # Example number of supported claims
support_score = supported_claims / total_claims

# Combine Scores
alpha, beta = 0.7, 0.3
overall_score = alpha * evidence_score + beta * support_score

print(f"Overall Evidence and Support Score: {overall_score}")
```
3. Confidence
    Sentiment Analysis: Use sentiment analysis tools to gauge the overall sentiment of the speech. Confidence often correlates with positive sentiment.
    Libraries: TextBlob, VADER, NLTK, spaCy
    Lexical Features: Identify and count the use of confident words and phrases.
    Example: Words like "definitely", "certainly", "undoubtedly", etc.


- Sentiment Score: Calculate the sentiment score using a sentiment analysis tool.
    ```
    from textblob import TextBlob

    def get_sentiment_score(text):
        blob = TextBlob(text)
        return blob.sentiment.polarity
    ```
- Lexical Confidence Score: Create a list of confident words and calculate their frequency in the speech.
    ```
    confident_words = ["definitely", "certainly", "undoubtedly", "surely", "confident"]

    def lexical_confidence_score(text):
        words = text.split()
        confident_count = sum(1 for word in words if word in confident_words)
        return confident_count / len(words)
    ```
- Prosodic Confidence Score: If you have prosodic data, you can use statistical measures to score confidence.
    ```
    def prosodic_confidence_score(pitch, volume, speech_rate):
        # Example formula: higher pitch, volume, and moderate speech rate indicate confidence
        return (pitch * 0.4) + (volume * 0.4) + (speech_rate * 0.2)
    ```
### Conclusion
1. Logical Coherence
- Identify Key Components
Identify the key components of logical coherence:

Claims: Statements or assertions.
Evidence: Supporting data or arguments.
Reasoning: Logical connections between claims and evidence.
- Dependency Parsing
Use dependency parsing to understand the grammatical structure of sentences and identify relationships between words. This helps in understanding how different parts of the speech are connected.
```
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp(speech_text)

for token in doc:
    print(f"{token.text} -> {token.dep_} -> {token.head.text}")
```
-  Argument Structure Analysis
Analyze the argument structure to ensure that claims are supported by evidence and reasoning:

Claim-Evidence Pairing: Check if each claim is followed by relevant evidence.
Reasoning Chains: Ensure that the reasoning logically connects the claims and evidence.
- Coherence Scoring
Develop a scoring system based on the identified components and their relationships:

Claim-Evidence Match: Score based on the presence and relevance of evidence for each claim.
Logical Flow: Score based on the logical progression of ideas.
Consistency: Score based on the consistency of arguments throughout the speech.

2. Conciseness
- Word Count:

Measure the total number of words spoken.
Formula: word_count = len(text.split())
- Sentence Length:

Calculate the average length of sentences.
Formula: average_sentence_length = sum(len(sentence.split()) for sentence in sentences) / len(sentences)
- Redundancy:

Identify and count repeated words or phrases.
Use techniques like n-gram analysis to detect repetition.
Formula: redundancy_score = count_repeated_phrases(text) / word_count
- Filler Words:

Detect and count filler words (e.g., "um", "uh", "like").
Maintain a list of common filler words and match them against the speech.
Formula: filler_word_count = sum(text.count(filler) for filler in filler_words_list)
- Information Density:

Measure the ratio of content words (nouns, verbs, adjectives, adverbs) to total words.
Use part-of-speech tagging to identify content words.
Formula: information_density = content_word_count / word_count
- Relevance:
Assess the relevance of each sentence to the main topic.
Use topic modeling or similarity measures (e.g., cosine similarity with the main topic vector).
Formula: relevance_score = sum(sentence_relevance_scores) / len(sentences)
Lexical Diversity:

- Calculate the variety of unique words used.
Formula: lexical_diversity = len(set(text.split())) / word_count
Clarity and Directness:

- Evaluate the use of clear and direct language.
Use readability scores (e.g., Flesch-Kincaid) or custom clarity metrics.
Formula: clarity_score = readability_score(text)

3. Impactfulness



## Project
- [Sequence-to-Sequence Generative Dialogue Systems](https://github.com/vliu15/dialogue-seq2seq)
- [Extraction and use of arguments in Recommender Systems](https://github.com/argrecsys/arg-miner)
- [ArgMiner: End-to-end Argument Mining](https://github.com/namiyousef/argument-mining)