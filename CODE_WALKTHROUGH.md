# Technical Code Walkthrough

**Tool:** Multi-Bank Earnings Call Sentiment Analysis  
**Client:** Scotiabank - Corporate Strategy and IT Teams  
**Prepared by:** MGSC 661 Consulting Team  
**Date:** December 2025  
**Purpose:** Detailed technical documentation for code understanding, maintenance, and customization

## Document Purpose

This document provides a comprehensive walkthrough of the sentiment analysis tool's code structure, logic, and implementation. It is intended for:

- **Data Scientists** who will maintain and enhance the tool
- **IT Support Staff** who will troubleshoot technical issues
- **Strategy Analysts** who want to understand the methodology at a deeper level
- **Future Development Teams** who may extend functionality

## Code Architecture Overview

The tool is structured as a linear Jupyter Notebook with 9 major steps:

```
Step 1: Configuration        → Define banks and URLs
Step 2: Import Libraries     → Load Python dependencies
Step 3-5: Load Resources     → Dictionary, stopwords, FinBERT model
Step 6-7: Define Functions   → Processing and analysis logic
Step 8: Process All Banks    → Main execution loop
Step 9: Output Results       → Rankings, visualization, examples
```

**Design Principles:**

1. **Sequential Processing:** Each step builds on previous steps
2. **Modular Functions:** Reusable functions for common operations
3. **Minimal Output:** Silent processing with final summary results
4. **Data Preservation:** All intermediate results stored for debugging
5. **Error Tolerance:** Continues processing even if individual sentences fail

## Detailed Step-by-Step Walkthrough

### Step 1: Bank Configuration

**Location:** Cell 1  
**Purpose:** Define which banks to analyze and their transcript URLs

```python
banks_to_analyze = [
    {'name': 'Scotiabank', 'url': 'https://...'},
    {'name': 'RBC', 'url': 'https://...'},
]
```

**Key Variables:**
- `banks_to_analyze` (list of dicts): Each dictionary contains bank name and PDF URL

**Customization Points:**
- Add more banks by appending to the list
- Change URLs for different quarters
- Add metadata fields (e.g., ticker symbol, fiscal year end)

**Design Rationale:**
Configuration is separated to allow non-technical users to update banks without modifying code.

---

### Step 2: Import Libraries

**Location:** Cell 2  
**Purpose:** Load all required Python packages

**Core Libraries:**

**Data Processing:**
- `pandas`: Tabular data manipulation
- `numpy`: Numerical operations

**Visualization:**
- `matplotlib.pyplot`: Chart creation
- `seaborn`: Statistical visualization styling

**Text Processing:**
- `re`: Regular expressions for pattern matching
- `collections.Counter`: Frequency counting

**PDF and Web:**
- `requests`: HTTP requests for PDF downloads
- `pypdf.PdfReader`: PDF text extraction
- `pathlib.Path`: File system operations

**Machine Learning:**
- `transformers`: Hugging Face library for FinBERT
- `torch`: PyTorch backend for neural networks

**Error Handling:**
If imports fail, verify installation with `pip list` and compare against requirements.txt.

---

### Step 3: Load Loughran-McDonald Dictionary

**Location:** Cell 3  
**Purpose:** Load financial sentiment word classifications

```python
lm_dict = pd.read_csv("Loughran-McDonald_MasterDictionary_1993-2024.csv")
lm_dict['word_lower'] = lm_dict['Word'].str.lower()
positive_words_set = set(lm_dict[lm_dict['Positive'] != 0]['word_lower'].tolist())
negative_words_set = set(lm_dict[lm_dict['Negative'] != 0]['word_lower'].tolist())
```

**Processing Logic:**

1. **Load CSV:** Dictionary contains ~90,000 words with sentiment classifications
2. **Normalize Case:** Convert all words to lowercase for case-insensitive matching
3. **Filter Positive:** Extract words where Positive column ≠ 0
4. **Filter Negative:** Extract words where Negative column ≠ 0
5. **Create Sets:** Use sets for O(1) lookup performance

**Data Structure:**
- `positive_words_set` (set): ~350 positive financial terms
- `negative_words_set` (set): ~2,350 negative financial terms

**Why Sets?**
Sets provide constant-time membership testing. Checking if a word is in a set is much faster than checking a list, critical when processing thousands of words.

**Customization:**
To add custom positive/negative words:
```python
positive_words_set.add('customword')
negative_words_set.add('customnegativeword')
```

---

### Step 4: Load Stopwords

**Location:** Cell 4  
**Purpose:** Load common words to exclude from analysis

```python
with open("stopwords.txt", 'r') as f:
    stopwords_set = set(word.strip().lower() for word in f.readlines())
```

**Processing Logic:**

1. **Read File:** Each line contains one stopword
2. **Strip Whitespace:** Remove newlines and spaces
3. **Lowercase:** Normalize for matching
4. **Store as Set:** Fast lookup during filtering

**Typical Stopwords:**
"the", "and", "is", "are", "was", "were", "a", "an", "to", "of", "in", "for"

**Why Remove Stopwords?**
These words appear frequently but carry no sentiment. Removing them:
- Reduces noise in word frequency analysis
- Speeds up processing
- Focuses analysis on meaningful content words

---

### Step 5: Load FinBERT Model

**Location:** Cell 5  
**Purpose:** Initialize the AI sentiment analysis model

```python
finbert_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
finbert_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
```

**Technical Details:**

**Tokenizer:**
- Converts text into numerical tokens the model can process
- Handles special tokens ([CLS], [SEP], [PAD])
- Max sequence length: 512 tokens (~400 words)

**Model:**
- Pre-trained BERT architecture fine-tuned on financial text
- Input: Tokenized text
- Output: Logits for 3 classes [positive, negative, neutral]
- Model size: ~440MB
- Parameters: ~110 million

**First-Time Execution:**
On first run, downloads model from Hugging Face (5-10 minutes). Subsequent runs load from cache.

**Memory Requirements:**
- Model loading: ~1GB RAM
- Inference: ~2GB RAM during processing

---

### Step 6: Text Processing Functions

**Location:** Cell 6  
**Purpose:** Define utility functions for text cleaning and preparation

#### Function: `clean_financial_text(text)`

**Purpose:** Normalize text for dictionary analysis

```python
def clean_financial_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = ' '.join(text.split())
    return text
```

**Processing Steps:**
1. Convert to lowercase
2. Remove all non-alphabetic characters (numbers, punctuation)
3. Collapse multiple spaces into single space
4. Return cleaned text

**Example:**
```
Input:  "Revenue grew 15.3% year-over-year!!"
Output: "revenue grew year over year"
```

**Why This Matters:**
Dictionary matching requires exact word matches. Cleaning ensures "grew!" matches "grew" in dictionary.

---

#### Function: `remove_stopwords(text, stopword_set)`

**Purpose:** Filter out common words without sentiment

```python
def remove_stopwords(text, stopword_set):
    words = text.split()
    filtered = [w for w in words if w not in stopword_set]
    return ' '.join(filtered)
```

**Processing Steps:**
1. Split text into individual words
2. Keep only words not in stopword set
3. Rejoin into cleaned text

**Example:**
```
Input:  "the revenue grew in the quarter"
Output: "revenue grew quarter"
```

**Performance:**
List comprehension with set lookup is optimized for speed. Processes ~100,000 words/second.

---

#### Function: `remove_boilerplate_text(text)`

**Purpose:** Remove page headers, footers, and metadata

```python
def remove_boilerplate_text(text):
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue
        if re.search(r'Page\s+\d+\s+of\s+\d+', line_stripped, re.IGNORECASE):
            continue
        # ... additional filters ...
        cleaned_lines.append(line_stripped)
    
    return ' '.join(cleaned_lines)
```

**Filtering Rules:**

1. **Empty Lines:** Skip blank lines
2. **Page Numbers:** Remove "Page X of Y" patterns
3. **Transcript Headers:** Remove "Earnings Call Transcript" metadata
4. **Section Headers:** Remove short all-caps lines (e.g., "Q&A SESSION")
5. **Speaker Labels:** Remove "Name - Company - Title" patterns

**Regular Expressions Used:**

- `r'Page\s+\d+\s+of\s+\d+'`: Matches page numbers
- `\s+`: One or more whitespace characters
- `\d+`: One or more digits

**Why Line-by-Line Processing?**
Boilerplate typically appears on separate lines. Line-by-line filtering is more accurate than whole-text patterns.

---

#### Function: `find_presentation_start(df)`

**Purpose:** Identify where substantive content begins

```python
def find_presentation_start(df):
    for idx, row in df.iterrows():
        text_upper = row['raw_text'].upper().replace(' ', '')
        if 'PRESENTATION' in text_upper or 'MANAGEMENTDISCUSSION' in text_upper:
            if 'PRESENTATION' in text_upper[:500] or 'MANAGEMENTDISCUSSION' in text_upper[:500]:
                if row['page_number'] > 1:
                    return row['page_number']
    return 1
```

**Detection Logic:**

1. Convert text to uppercase and remove spaces (handles "PRESENTATION" or "P R E S E N T A T I O N")
2. Check if "PRESENTATION" or "MANAGEMENTDISCUSSION" appears
3. Verify it appears in first 500 characters (top of page, not buried in text)
4. Ensure page number > 1 (don't skip page 1 accidentally)
5. Return page number where content starts

**Fallback:**
If no marker found, returns page 1 (analyze entire document).

**Why This Matters:**
First 1-3 pages typically contain:
- Cover page with legal disclaimers
- Forward-looking statement warnings
- Participant lists

These pages dilute sentiment analysis with legal boilerplate.

---

### Step 7: Analysis Functions

**Location:** Cell 7  
**Purpose:** Define core sentiment analysis logic

#### Function: `calculate_sentiment(text, pos_set, neg_set)`

**Purpose:** Dictionary-based sentiment calculation

```python
def calculate_sentiment(text, pos_set, neg_set):
    words = text.split()
    pos_count = sum(1 for w in words if w in pos_set)
    neg_count = sum(1 for w in words if w in neg_set)
    net_sent = pos_count - neg_count
    return pos_count, neg_count, net_sent
```

**Algorithm:**

1. Split text into words
2. Count words in positive set
3. Count words in negative set
4. Calculate net sentiment (positive - negative)
5. Return all three values

**Complexity:** O(n) where n is number of words. Each word requires one set lookup (O(1)).

**Example:**
```
Text: "strong performance despite challenges regulatory concerns"
Positive words: strong, performance = 2
Negative words: challenges, concerns = 2
Net sentiment: 2 - 2 = 0
```

**Return Format:**
Tuple of three integers: (positive_count, negative_count, net_sentiment)

---

#### Function: `analyze_with_finbert(text, tokenizer, model)`

**Purpose:** AI-based contextual sentiment analysis

```python
def analyze_with_finbert(text, tokenizer, model):
    tokenized = tokenizer(text, return_tensors="pt", truncation=True, 
                         max_length=512, padding=True)
    with torch.no_grad():
        outputs = model(**tokenized)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    pos_prob = probs[0][0].item()
    neg_prob = probs[0][1].item()
    neu_prob = probs[0][2].item()
    
    if pos_prob > neg_prob and pos_prob > neu_prob:
        label = 'positive'
    elif neg_prob > pos_prob and neg_prob > neu_prob:
        label = 'negative'
    else:
        label = 'neutral'
    
    net_score = pos_prob - neg_prob
    return pos_prob, neg_prob, neu_prob, label, net_score
```

**Processing Pipeline:**

1. **Tokenization:**
   - Convert text to model-compatible format
   - Truncate to 512 tokens if needed
   - Add padding to standardize length
   - Return as PyTorch tensor

2. **Model Inference:**
   - `torch.no_grad()`: Disable gradient calculation (faster inference)
   - Forward pass through model
   - Get raw logits (unnormalized scores)

3. **Probability Calculation:**
   - Apply softmax to convert logits to probabilities
   - Sum of probabilities = 1.0
   - Extract individual class probabilities

4. **Classification:**
   - Label = class with highest probability
   - Handles ties by prioritizing: positive > negative > neutral

5. **Net Score:**
   - net_score = positive_prob - negative_prob
   - Range: -1.0 to +1.0

**Technical Notes:**

- `return_tensors="pt"`: Return PyTorch tensors
- `truncation=True`: Handle texts longer than 512 tokens
- `max_length=512`: BERT's maximum sequence length
- `padding=True`: Pad shorter texts to batch length

**Why torch.no_grad()?**
Gradient calculation is only needed for training. Disabling it during inference saves memory and speeds up processing.

---

#### Function: `analyze_sentence_finbert(sentence, tokenizer, model)`

**Purpose:** Analyze individual sentence with validation

```python
def analyze_sentence_finbert(sentence, tokenizer, model):
    sentence = sentence.strip()
    
    # Validation checks
    if len(sentence) < 20:
        return None
    if re.search(r'Page\s+\d+\s+of\s+\d+', sentence, re.IGNORECASE):
        return None
    if 'Earnings Call Transcript' in sentence:
        return None
    if len(sentence) < 60 and sentence.count(' ') < 5:
        return None
    
    try:
        # Same tokenization and inference as analyze_with_finbert
        tokenized = tokenizer(sentence, return_tensors="pt", truncation=True,
                             max_length=512, padding=True)
        with torch.no_grad():
            outputs = model(**tokenized)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        return {
            'sentence': sentence[:300],
            'positive_prob': probs[0][0].item(),
            'negative_prob': probs[0][1].item(),
            'net_score': probs[0][0].item() - probs[0][1].item()
        }
    except:
        return None
```

**Validation Logic:**

1. **Minimum Length:** Sentences < 20 characters likely fragments
2. **Page Numbers:** Skip boilerplate like "Page 5 of 17"
3. **Headers:** Skip metadata lines
4. **Word Count:** Short sentences with few spaces likely labels not content

**Error Handling:**
Try-except catches tokenization errors (e.g., unusual characters). Returns None instead of crashing.

**Return Format:**
Dictionary with sentence text and scores, or None if invalid.

**Why Return None?**
Allows calling code to filter out invalid sentences with simple `if result` checks.

---

### Step 8: Main Processing Loop

**Location:** Cell 8  
**Purpose:** Execute analysis for all configured banks

This is the largest code block. We'll break it down into logical sections.

#### Section A: Initialization

```python
all_results = []
all_bank_data = {}

print("Processing banks...\n")

for i, bank_config in enumerate(banks_to_analyze, 1):
    bank_name = bank_config['name']
    bank_url = bank_config['url']
    
    print(f"{i}/{len(banks_to_analyze)}: {bank_name}")
```

**Data Structures:**

- `all_results` (list): Stores summary statistics for each bank
- `all_bank_data` (dict): Stores full dataframes for each bank

**Output:**
Minimal progress indicator showing "1/3: Scotiabank", "2/3: RBC", etc.

---

#### Section B: PDF Download and Text Extraction

```python
# Download PDF
pdf_filename = f"{bank_name.replace(' ', '_')}_transcript.pdf"
response = requests.get(bank_url, timeout=60)
response.raise_for_status()
Path(pdf_filename).write_bytes(response.content)

# Extract text
reader = PdfReader(pdf_filename)
pages_data = []
for page_num, page in enumerate(reader.pages, start=1):
    pages_data.append({
        'page_number': page_num,
        'raw_text': page.extract_text()
    })

df = pd.DataFrame(pages_data)
```

**Download Process:**

1. **Generate Filename:** Replace spaces with underscores
2. **HTTP GET Request:** 60-second timeout for large PDFs
3. **Error Checking:** `raise_for_status()` throws exception on 404, 500, etc.
4. **Write to Disk:** Store PDF locally

**Text Extraction:**

1. **Create Reader:** Initialize PDF parser
2. **Iterate Pages:** Process each page sequentially
3. **Extract Text:** PDF library handles text extraction
4. **Store with Page Number:** Maintain page context

**Dataframe Structure:**
```
   page_number  raw_text
0  1            [Page 1 content]
1  2            [Page 2 content]
...
```

**Error Points:**
- Network failure: Timeout or connection error
- Invalid URL: 404 error
- Corrupted PDF: Extraction failure

---

#### Section C: Content Filtering and Cleaning

```python
# Find substantive content and clean
start_page = find_presentation_start(df)
df = df[df['page_number'] >= start_page].copy()
df['raw_text'] = df['raw_text'].apply(remove_boilerplate_text)
df['cleaned_text'] = df['raw_text'].apply(clean_financial_text)
df['cleaned_text'] = df['cleaned_text'].apply(lambda x: remove_stopwords(x, stopwords_set))
df['word_count'] = df['cleaned_text'].str.split().str.len()

full_raw_text = ' '.join(df['raw_text'].tolist())
full_cleaned_text = ' '.join(df['cleaned_text'].tolist())
```

**Processing Steps:**

1. **Find Start:** Identify first substantive page
2. **Filter DataFrame:** Keep only pages >= start_page
3. **Remove Boilerplate:** Clean headers/footers from raw text
4. **Normalize Text:** Apply financial text cleaning
5. **Remove Stopwords:** Filter common words
6. **Count Words:** Calculate words per page
7. **Create Combined Text:** Join all pages for overall analysis

**Column Evolution:**
```
Initial:   page_number, raw_text
After C:   page_number, raw_text, cleaned_text, word_count
```

**Why .copy()?**
`df[df['page_number'] >= start_page].copy()` creates independent copy, avoiding SettingWithCopyWarning.

---

#### Section D: Dictionary Analysis

```python
# Dictionary analysis
sentiment_results = df['cleaned_text'].apply(
    lambda text: calculate_sentiment(text, positive_words_set, negative_words_set)
)
df['dict_pos_count'] = sentiment_results.apply(lambda x: x[0])
df['dict_neg_count'] = sentiment_results.apply(lambda x: x[1])
df['dict_net_sent'] = sentiment_results.apply(lambda x: x[2])
df['dict_sent_ratio'] = ((df['dict_net_sent'] / df['word_count']) * 100).fillna(0)

dict_total_pos = df['dict_pos_count'].sum()
dict_total_neg = df['dict_neg_count'].sum()
dict_total_words = df['word_count'].sum()
dict_overall_ratio = ((dict_total_pos - dict_total_neg) / dict_total_words) * 100
```

**Page-Level Analysis:**

1. Apply `calculate_sentiment` to each page's cleaned text
2. Extract tuple components into separate columns:
   - `dict_pos_count`: Positive words per page
   - `dict_neg_count`: Negative words per page
   - `dict_net_sent`: Net sentiment per page
3. Calculate ratio: (net sentiment / word count) × 100
4. Handle division by zero with `.fillna(0)`

**Overall Analysis:**

1. Sum positive words across all pages
2. Sum negative words across all pages
3. Sum total words across all pages
4. Calculate overall ratio: (net / total) × 100

**Example Calculation:**
```
Total positive words: 150
Total negative words: 100
Total words: 10,000

Overall ratio = (150 - 100) / 10,000 × 100 = +0.5%
```

---

#### Section E: FinBERT Analysis

```python
# FinBERT analysis
finbert_results = []
for idx, row in df.iterrows():
    pos, neg, neu, lbl, net = analyze_with_finbert(
        row['raw_text'], finbert_tokenizer, finbert_model
    )
    finbert_results.append({
        'page_number': row['page_number'],
        'finbert_pos_prob': pos,
        'finbert_neg_prob': neg,
        'finbert_neu_prob': neu,
        'finbert_label': lbl,
        'finbert_net_score': net
    })

finbert_df = pd.DataFrame(finbert_results)
df = df.merge(finbert_df, on='page_number', how='left')

finbert_avg_net = df['finbert_net_score'].mean()
finbert_label_counts = df['finbert_label'].value_counts()
```

**Processing Flow:**

1. **Initialize Results List:** Store page-level results
2. **Iterate Pages:** Process each page's raw text
3. **Analyze with FinBERT:** Get probabilities and label
4. **Append Results:** Add to list as dictionary
5. **Create DataFrame:** Convert list of dicts to DataFrame
6. **Merge:** Join FinBERT results back to main dataframe

**Why Use Raw Text for FinBERT?**
FinBERT benefits from complete context including punctuation and capitalization. Dictionary method needs normalized text; FinBERT uses raw text.

**Aggregation:**

- `finbert_avg_net`: Mean of all page net scores
- `finbert_label_counts`: Count of positive/negative/neutral pages

**Performance Note:**
This is the slowest part of processing. Each page requires neural network inference (~0.5-1 second per page).

---

#### Section F: Example Sentence Extraction

```python
# Find example sentences
most_pos_page = df.nlargest(1, 'finbert_net_score').iloc[0]
most_neg_page = df.nsmallest(1, 'finbert_net_score').iloc[0]

pos_sentences = []
for sent in most_pos_page['raw_text'].split('.'):
    result = analyze_sentence_finbert(sent, finbert_tokenizer, finbert_model)
    if result:
        pos_sentences.append(result)
    if len(pos_sentences) >= 5:
        break

neg_sentences = []
for sent in most_neg_page['raw_text'].split('.'):
    result = analyze_sentence_finbert(sent, finbert_tokenizer, finbert_model)
    if result:
        neg_sentences.append(result)
    if len(neg_sentences) >= 5:
        break
```

**Selection Logic:**

1. **Find Extreme Pages:**
   - Most positive: Page with highest net score
   - Most negative: Page with lowest net score

2. **Extract Sentences:**
   - Split page text by periods
   - Analyze each sentence individually
   - Keep valid sentences (passes validation)
   - Stop after collecting 5 candidates

3. **Find Best Examples:**
   - Not shown: Later code selects top 3 from candidates

**Why Split by Period?**
Simple sentence boundary detection. More sophisticated methods (spaCy, NLTK) possible but add dependencies.

**Why Analyze Top 5?**
Ensures we have enough candidates to find true best sentences (some may be filtered out).

---

#### Section G: Store Results

```python
# Store results
all_results.append({
    'bank': bank_name,
    'dict_score': dict_overall_ratio,
    'finbert_score': finbert_avg_net,
    'dict_pos_words': dict_total_pos,
    'dict_neg_words': dict_total_neg,
    'finbert_pos_pages': finbert_label_counts.get('positive', 0),
    'finbert_neg_pages': finbert_label_counts.get('negative', 0),
    'total_pages': len(df)
})

all_bank_data[bank_name] = {'df': df}
```

**Summary Statistics:**

Each bank gets one entry in `all_results` with:
- Overall scores (both methods)
- Raw counts (words, pages)
- Metadata (total pages analyzed)

**Full Data Preservation:**

`all_bank_data` stores complete dataframe for each bank, enabling:
- Detailed sentence extraction later
- Page-level drill-down
- Custom analysis by users

---

### Step 9: Results Output

**Location:** Cells 9-12  
**Purpose:** Display comparative results and export data

#### Cell 9: Summary Table

```python
comparison_df = pd.DataFrame(all_results)
comparison_df['dict_rank'] = comparison_df['dict_score'].rank(ascending=False).astype(int)
comparison_df['finbert_rank'] = comparison_df['finbert_score'].rank(ascending=False).astype(int)

display_df = comparison_df[[
    'bank', 'dict_score', 'dict_rank', 'finbert_score', 'finbert_rank'
]].sort_values('finbert_score', ascending=False)

print("\nSentiment Rankings:\n")
print(display_df.to_string(index=False))
```

**Ranking Logic:**

1. Create DataFrame from results list
2. Calculate ranks using `.rank(ascending=False)`:
   - Higher score = Lower rank number (rank 1 is best)
   - Automatically handles ties
3. Select key columns for display
4. Sort by FinBERT score (highest first)
5. Print without row indices

**Output Format:**
```
bank         dict_score  dict_rank  finbert_score  finbert_rank
RBC          +1.85%      1          +0.463         1
Scotiabank   +1.42%      2          +0.351         2
```

---

#### Cell 10: Comparative Visualization

```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Dictionary comparison
sorted_dict = comparison_df.sort_values('dict_score', ascending=False)
x = range(len(sorted_dict))
colors_dict = ['#06A77D' if s > 0 else '#E63946' for s in sorted_dict['dict_score']]

bars1 = ax1.bar(x, sorted_dict['dict_score'], color=colors_dict,
               edgecolor='black', linewidth=2, width=0.6)
ax1.set_xticks(x)
ax1.set_xticklabels(sorted_dict['bank'])
ax1.axhline(y=0, color='black', linewidth=2)
ax1.set_ylabel('Sentiment Ratio (%)', fontweight='bold')
ax1.set_title('Dictionary Method: Bank Comparison', fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')

for bar, score in zip(bars1, sorted_dict['dict_score']):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, height,
            f'{score:+.2f}%', ha='center',
            va='bottom' if height > 0 else 'top',
            fontweight='bold', fontsize=11)
```

**Visualization Components:**

**Layout:**
- `subplots(1, 2)`: Two charts side-by-side
- `figsize=(16, 6)`: Wide format for readability

**Color Logic:**
- Green (#06A77D) for positive scores
- Red (#E63946) for negative scores
- Conditional list comprehension for dynamic coloring

**Bar Chart Elements:**
- `edgecolor='black'`: Black borders for definition
- `linewidth=2`: Thick borders for professional look
- `width=0.6`: Narrower bars for visual clarity

**Reference Line:**
- `axhline(y=0)`: Horizontal line at zero
- Makes positive/negative distinction clear

**Labels:**
- Positioned dynamically based on bar height
- `va='bottom'/'top'`: Label above positive bars, below negative
- Formatted with `+` sign for positive values

**Axis Formatting:**
- Grid for easy value reading
- Bold labels for emphasis
- Descriptive title with methodology name

**FinBERT Chart:**
Similar structure but with FinBERT scores and adjusted scale (-1 to +1).

---

#### Cell 11: Example Sentences

```python
for bank_name in comparison_df.sort_values('finbert_score', ascending=False)['bank']:
    print(f"\n{bank_name}")
    print()
    
    bank_df = all_bank_data[bank_name]['df']
    
    # Get top 3 positive pages
    top_pos_pages = bank_df.nlargest(3, 'finbert_net_score')
    
    print("Top 3 Positive Sentences:")
    pos_count = 0
    for _, page in top_pos_pages.iterrows():
        for sent in page['raw_text'].split('.'):
            result = analyze_sentence_finbert(sent, finbert_tokenizer, finbert_model)
            if result and result['net_score'] > 0:
                print(f"  {pos_count + 1}. \"{result['sentence']}\"")
                print(f"     Net: {result['net_score']:+.3f}")
                print()
                pos_count += 1
                if pos_count >= 3:
                    break
        if pos_count >= 3:
            break
```

**Extraction Strategy:**

1. **Sort Banks:** Start with most positive bank
2. **Get Top Pages:** Select 3 pages with highest FinBERT scores
3. **Extract Sentences:** Split pages by period
4. **Filter Positive:** Keep only sentences with positive net score
5. **Limit Results:** Stop after finding 3 sentences
6. **Format Output:** Display sentence text and score

**Nested Loop Logic:**
- Outer loop: Iterate through top pages
- Inner loop: Iterate through sentences in page
- Break conditions: Stop when 3 sentences found

**Why Top 3 Pages?**
Ensures diverse examples. If we only looked at one page, all 3 sentences might be similar.

**Similar Logic for Negative Sentences:**
Same structure but filters for negative scores and uses `nsmallest()`.

---

#### Cell 12: Export Results

```python
output_file = 'multi_bank_sentiment_comparison.csv'
comparison_df.to_csv(output_file, index=False)
print(f"Results exported to: {output_file}")
```

**Export Format:**
CSV with columns: bank, dict_score, finbert_score, dict_pos_words, dict_neg_words, finbert_pos_pages, finbert_neg_pages, total_pages, dict_rank, finbert_rank

**Use Cases:**
- Import into Excel for further analysis
- Create PowerPoint charts
- Store for historical comparison
- Feed into business intelligence tools

---

## Performance Characteristics

### Processing Time

**Per Bank (approximate):**
- PDF download: 5-10 seconds
- Text extraction: 2-5 seconds
- Dictionary analysis: 1-2 seconds
- FinBERT analysis: 30-60 seconds (depends on page count)
- Example extraction: 5-10 seconds
- **Total: ~1-2 minutes per bank**

**Bottleneck:** FinBERT inference on CPU. Can be 5-10x faster with GPU.

### Memory Usage

**Peak Memory:**
- Base Python + libraries: ~500 MB
- FinBERT model: ~1 GB
- Processing one bank: ~200 MB
- **Total: ~2 GB recommended**

**Memory Optimization:**
Process banks sequentially rather than loading all at once.

### Disk Space

**Temporary:**
- Downloaded PDFs: ~500 KB - 2 MB each
- Can be deleted after processing

**Permanent:**
- CSV export: < 10 KB
- FinBERT model cache: ~440 MB (one-time)

---

## Customization Guide

### Adding Custom Sentiment Words

```python
# After Step 3
positive_words_set.add('innovative')
positive_words_set.add('breakthrough')
negative_words_set.add('headwinds')
```

### Adjusting Boilerplate Filters

```python
# In remove_boilerplate_text function
# Add custom pattern
if 'Your Custom Pattern' in line_stripped:
    continue
```

### Changing Number of Example Sentences

```python
# In Step 8, Section F
if len(pos_sentences) >= 5:  # Change to 10 for more candidates
    break
```

### Modifying Visualization Colors

```python
# In Step 9, Cell 10
POSITIVE_COLOR = '#06A77D'  # Current green
NEGATIVE_COLOR = '#E63946'  # Current red
# Change these constants at top of cell
```

### Exporting Additional Formats

```python
# After Step 9, Cell 12
comparison_df.to_excel('results.xlsx', index=False)
comparison_df.to_json('results.json', orient='records')
```

---

## Debugging Guide

### Enable Verbose Output

Add print statements in processing loop:

```python
# In Step 8, after each major section
print(f"  Downloaded: {len(df)} pages")
print(f"  Dictionary: {dict_overall_ratio:.2f}%")
print(f"  FinBERT: {finbert_avg_net:.3f}")
```

### Inspect Intermediate Data

```python
# After processing
print(df.head())  # View first few rows
print(df.columns)  # View all columns
print(df.describe())  # Statistical summary
```

### Test Individual Functions

```python
# Test text cleaning
sample_text = "Example text with $numbers$ and punctuation!!"
cleaned = clean_financial_text(sample_text)
print(f"Original: {sample_text}")
print(f"Cleaned: {cleaned}")
```

### Validate FinBERT Scores

```python
# Check if probabilities sum to 1
for idx, row in df.iterrows():
    total = row['finbert_pos_prob'] + row['finbert_neg_prob'] + row['finbert_neu_prob']
    if abs(total - 1.0) > 0.01:
        print(f"Warning: Page {row['page_number']} probs don't sum to 1: {total}")
```

---

## Error Handling

### Network Errors

```python
# Wrap PDF download in try-except
try:
    response = requests.get(bank_url, timeout=60)
    response.raise_for_status()
except requests.exceptions.RequestException as e:
    print(f"Failed to download {bank_name}: {e}")
    continue  # Skip this bank
```

### PDF Extraction Errors

```python
# Wrap extraction in try-except
try:
    reader = PdfReader(pdf_filename)
    pages_data = [...]
except Exception as e:
    print(f"Failed to extract text from {bank_name}: {e}")
    continue
```

### FinBERT Inference Errors

Already handled in `analyze_sentence_finbert()` with try-except returning None.

---

## Security Considerations

### Input Validation

**URLs:** Only accept HTTPS URLs from known domains
```python
if not bank_url.startswith('https://'):
    raise ValueError("Only HTTPS URLs allowed")
```

**File Paths:** Validate PDF filenames to prevent directory traversal
```python
import os
safe_filename = os.path.basename(pdf_filename)
```

### Data Privacy

**Temporary Files:** Consider deleting PDFs after processing
```python
import os
os.remove(pdf_filename)
```

**Sensitive Information:** Earnings transcripts are public, but be cautious about internal notes added to analysis

---

## Extension Opportunities

### Historical Trend Analysis

Store results in database with timestamps:
```python
comparison_df['quarter'] = 'Q1 2025'
comparison_df['analysis_date'] = pd.Timestamp.now()
comparison_df.to_sql('sentiment_history', conn, if_exists='append')
```

### Automated Scheduling

Use cron (Linux/Mac) or Task Scheduler (Windows):
```bash
# Run every Monday at 9 AM
0 9 * * 1 /usr/bin/python3 /path/to/notebook_runner.py
```

### API Integration

Convert to Flask API:
```python
from flask import Flask, request, jsonify

@app.route('/analyze', methods=['POST'])
def analyze():
    bank_config = request.json
    # Run analysis
    return jsonify(results)
```

### Real-Time Monitoring

Add email alerts for significant sentiment changes:
```python
if abs(current_score - previous_score) > 0.2:
    send_email_alert(bank_name, current_score, previous_score)
```

---

## Maintenance Schedule

### Weekly
- Monitor for any processing errors
- Verify PDF URLs still accessible

### Monthly
- Review example sentences for quality
- Check for FinBERT model updates

### Quarterly
- Run analysis on new transcripts
- Archive results
- Update documentation with findings

### Annually
- Update Loughran-McDonald dictionary (January release)
- Review and update boilerplate filters
- Assess need for methodology enhancements

---

## Appendix: Key Algorithms

### Sentiment Ratio Calculation

```
Dictionary Score = ((Positive Words - Negative Words) / Total Words) × 100

Example:
Positive: 150 words
Negative: 100 words
Total: 10,000 words
Score = (150 - 100) / 10,000 × 100 = +0.5%
```

### FinBERT Net Score

```
For each page:
  Net Score = Positive Probability - Negative Probability

Overall Score = Average of all page net scores

Example:
Page 1: 0.75 - 0.10 = +0.65
Page 2: 0.60 - 0.15 = +0.45
Page 3: 0.55 - 0.20 = +0.35
Average = (0.65 + 0.45 + 0.35) / 3 = +0.48
```

### Ranking Formula

```
Rank = Position when sorted descending by score
Highest score = Rank 1
Ties receive same rank
```

---

**Document Version:** 1.0  
**Last Updated:** December 2025  
**Maintained by:** MGSC 661 Consulting Team  
**Questions:** Contact technical lead for clarifications
