# BTC1889H – Deep Learning
# Winter Term 2026
# Team Project
# Team 1: Viroosh Brendra, Agata Wolochacz, Lauren MacIntyre, Yaline Ganesathasan, Burrak Urrehman

# Load packages
library(keras3)
library(stringr)

# PREPROCESSING -----------------------------------------------------------

# Set working directory (update path as needed)
setwd("~/Desktop/team_project/")

# ── 1. Load Raw Data ─────────────────────────────────────────
# df_train = training set | df_test = testing set
df_train <- read.csv("Corona_NLP_train.csv", stringsAsFactors = FALSE)
df_test  <- read.csv("Corona_NLP_test.csv",  stringsAsFactors = FALSE)

cat("Training samples:", nrow(df_train), "\n")
cat("Testing  samples:", nrow(df_test),  "\n")

cat("\nTraining sentiment distribution:\n")
print(table(df_train$Sentiment))

cat("\nTesting sentiment distribution:\n")
print(table(df_test$Sentiment))


# ── 2. Extract Raw Text and Labels ───────────────────────────
train_texts  <- df_train$OriginalTweet
test_texts   <- df_test$OriginalTweet
train_labels <- df_train$Sentiment
test_labels  <- df_test$Sentiment

# Tweets contain non-UTF-8 bytes (e.g. latin-1 special characters,
# emoji byte sequences) so converted each string to UTF-8, replacing any invalid bytes
# with a blank, then strip any remaining non-ASCII characters.
clean_encoding <- function(x) {
  x <- iconv(x, from = "", to = "UTF-8", sub = " ")   # replace bad bytes
  x <- iconv(x, from = "UTF-8", to = "ASCII", sub = " ") # drop non-ASCII
  x <- gsub("\\s+", " ", x)                            # collapse whitespace
  trimws(x)
}

train_texts <- clean_encoding(train_texts)
test_texts  <- clean_encoding(test_texts)


# ── 3. Explore Text Length (informs maxlen choice) ───────────
# Count words per tweet (approx, splitting on whitespace)
ll <- str_count(train_texts, "\\S+")

cat("\nTweet word-count summary (training set):\n")
print(summary(ll))
hist(ll, main = "Tweet word-count distribution (training)",
     xlab = "Number of words", col = "steelblue")

# What % of tweets fall within our chosen maxlen?3
# Following tutorial 9 approach of checking coverage
maxlen <- 50   # tweets are short; 50 covers the large majority
cat(sprintf("\nProportion of tweets with <= %d words: %.1f%%\n",
            maxlen, mean(ll <= maxlen) * 100))


# ── 4. Tokenization + Vectorization ──────────────────────────
# Following Tutorial 9: use layer_text_vectorization() + adapt()
# max_tokens = vocabulary size cap (top N most frequent words)
# output_mode = "int"  -> produces integer token ID sequences
# output_sequence_length = maxlen -> handles padding/truncation on-graph

max_words <- 10000   # vocabulary size; reasonable for tweet corpus

vec <- layer_text_vectorization(
  max_tokens = max_words,
  standardize = "lower_and_strip_punctuation",  # lowercases + strips punctuation
  split = "whitespace",
  output_mode = "int",
  output_sequence_length = maxlen               # pads/truncates to fixed length
)

# Fit vocabulary on TRAINING text only (avoid data leakage)
adapt(vec, train_texts)

# Inspect vocabulary
vocab <- vec$get_vocabulary()
cat("\nVocabulary size (including padding + OOV tokens):", length(vocab), "\n")
cat("Top 10 tokens:", paste(vocab[1:10], collapse = ", "), "\n")
# Note: index 0 = padding, index 1 = [UNK] (OOV), real words start at index 2


# ── 5. Produce Integer Sequences ─────────────────────────────
# Apply the fitted vectorizer to both splits
# Output is already padded/truncated to maxlen (as in Tutorial 9)
x_train <- as.array(vec(train_texts))   # shape: (n_train, maxlen)
x_test  <- as.array(vec(test_texts))    # shape: (n_test,  maxlen)

cat("\nShape of x_train:", paste(dim(x_train), collapse = " x "), "\n")
cat("Shape of x_test: ", paste(dim(x_test),  collapse = " x "), "\n")

# Example: first training tweet as integer sequence
cat("\nFirst training tweet (integer sequence):\n")
print(x_train[1, ])


# ── 6. Shuffle Training Data ─────────────────────────────────
# Following Tutorial 10: shuffle before training
set.seed(123)
I <- sample.int(nrow(x_train))
x_train      <- x_train[I, ]
train_labels <- train_labels[I]


# ── 7. Label Encoding ────────────────────────────────────────
# Five ordered sentiment classes -> integers 0-4 -> one-hot
# Ordinal order preserved: Extremely Negative (0) -> Extremely Positive (4)

sentiment_levels <- c(
  "Extremely Negative",   # 0
  "Negative",             # 1
  "Neutral",              # 2
  "Positive",             # 3
  "Extremely Positive"    # 4
)

NUM_CLASSES <- length(sentiment_levels)

# Integer encoding (0-indexed for Keras)
y_train_int <- match(train_labels, sentiment_levels) - 1L
y_test_int  <- match(test_labels,  sentiment_levels) - 1L

# No NAs means all labels were recognised
stopifnot(!any(is.na(y_train_int)), !any(is.na(y_test_int)))

# One-hot encode for categorical_crossentropy
y_train <- to_categorical(y_train_int, num_classes = NUM_CLASSES)
y_test  <- to_categorical(y_test_int,  num_classes = NUM_CLASSES)

cat("\nShape of y_train:", paste(dim(y_train), collapse = " x "), "\n")
cat("Shape of y_test: ", paste(dim(y_test),  collapse = " x "), "\n")

cat("\nLabel encoding:\n")
for (i in seq_along(sentiment_levels)) {
  cat(sprintf("  %d  ->  %s\n", i - 1L, sentiment_levels[i]))
}


# ── 8. Key Parameter Summary ─────────────────────────────────
cat("\n--- Preprocessing Parameters ---\n")
cat("max_words (vocabulary size) :", max_words,    "\n")
cat("maxlen (sequence length)    :", maxlen,       "\n")
cat("NUM_CLASSES                 :", NUM_CLASSES,  "\n")
cat("Training samples            :", nrow(x_train),"\n")
cat("Testing  samples            :", nrow(x_test), "\n")

