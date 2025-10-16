from dataclasses import InitVar, dataclass, field
from collections import Counter
import numpy as np
import structlog
import tensorflow_datasets as tfds

log = structlog.get_logger()


@dataclass
class Data:
    """Handles generation of tokenized data for MLP with trainable embeddings."""

    rng: InitVar[np.random.Generator]
    dataset_name: str
    vocab_size: int = 10000  # Size of vocabulary
    max_seq_length: int = 50  # Maximum sequence length
    train_text_set: np.ndarray = field(init=False)
    train_label_set: np.ndarray = field(init=False)
    test_text_set: np.ndarray = field(init=False)
    test_label_set: np.ndarray = field(init=False)
    train_index: np.ndarray = field(init=False)
    word_to_idx: dict = field(init=False)
    idx_to_word: dict = field(init=False)

    def __post_init__(self, rng: np.random.Generator) -> None:
        """Generate training and validating data."""

        # load data from TF
        (ag_train, ag_test) = tfds.load(
            self.dataset_name,
            split=["train", "test"],
            shuffle_files=True,
        )

        def combine_text(x):
            text = x["title"] + " " + x["description"]
            return text, x["label"]

        ag_train = ag_train.map(combine_text)
        ag_test = ag_test.map(combine_text)

        # Convert to numpy arrays
        train_texts = [x.numpy().decode('utf-8') for x, _ in ag_train]
        self.train_label_set = np.array([y for _, y in ag_train])
        test_texts = [x.numpy().decode('utf-8') for x, _ in ag_test]
        self.test_label_set = np.array([y for _, y in ag_test])

        log.debug(
            "Data initialized",
            train_samples=len(train_texts),
            test_samples=len(test_texts),
            train_labels=self.train_label_set.shape,
            test_labels=self.test_label_set.shape,
        )

        log.info("Building vocabulary")
        self.build_vocabulary(train_texts)
        
        # Tokenize and pad sequences
        log.info("Tokenizing text")
        self.train_text_set = self.tokenize_texts(train_texts)
        self.test_text_set = self.tokenize_texts(test_texts)
        self.train_index = np.arange(self.train_text_set.shape[0])

        log.debug(
            "Data processed",
            train_text_shape=self.train_text_set.shape,
            test_text_shape=self.test_text_set.shape,
            vocab_size=len(self.word_to_idx),
            max_seq_length=self.max_seq_length,
        )

    def build_vocabulary(self, texts: list) -> None:
        """Build vocabulary from training texts."""
        word_counts = Counter()
        
        for text in texts:
            words = text.lower().split()
            word_counts.update(words)
        
        most_common = word_counts.most_common(self.vocab_size - 2)
        
        # Create word to index mapping
        self.word_to_idx = {
            '<PAD>': 0,
            '<UNK>': 1,
        }
        
        for idx, (word, _) in enumerate(most_common, start=2):
            self.word_to_idx[word] = idx
        
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}

    def tokenize_texts(self, texts: list) -> np.ndarray:
        """Convert texts to padded token sequences."""
        tokenized = []
        
        for text in texts:
            words = text.lower().split()
            # Convert words to indices
            indices = [self.word_to_idx.get(word, 1) for word in words]  # 1 is <UNK>
            
            # Truncate or pad to max_seq_length
            if len(indices) > self.max_seq_length:
                indices = indices[:self.max_seq_length]
            else:
                indices.extend([0] * (self.max_seq_length - len(indices)))  # 0 is <PAD>
            
            tokenized.append(indices)
        
        return np.array(tokenized, dtype=np.int32)

    def get_batch(
        self, rng: np.random.Generator, batch_size: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Select random subset of examples for training batch. Not used in this assignment."""
        choices = rng.choice(self.train_index, size=batch_size)

        texts = self.train_text_set[choices]
        labels = self.train_label_set[choices].flatten()

        return texts, labels