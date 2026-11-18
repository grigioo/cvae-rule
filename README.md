# CVAE Procedural Dungeon Generator with Domain-Specific Rule Loss

A procedural level generation project for 2D roguelike games using a **Conditional Variational Autoencoder (CVAE)** trained on dungeon layouts from *The Legend of Zelda*. The project compares a standard CVAE architecture against a rule-guided CVAE that evaluates structural constraints (wall integrity, absence of internal voids, valid object placement, and unobstructed door passages).

## Project Structure

```text
cvae-rule-loss/
├── config.py                  # Centralized paths, tile mappings, and hyperparameters
├── main.py                    # CLI entry point for data prep, training, and generation
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
├── notebooks/
│   └── colab_runner.ipynb     # Lightweight Google Colab launcher
├── src/
│   ├── data/                  # Level slicing, deduplication, selection, augmentation, encoding
│   ├── models/                # Encoder/Decoder layers, rule loss, Base & Rule CVAE models
│   └── generation/            # Single room & full BFS-based dungeon graph generators
├── data/
│   └── raw/                   # Input ASCII level maps (e.g., tloz1_1.txt)
└── saved_models/              # Trained .keras model checkpoints
```

## Tile Vocabulary
Each room is represented as a 16 x 11 ASCII grid encoded into 10 one-hot channels:

| **Token** | **ID** | **Description**               |
| --------- | ------ | ----------------------------- |
| `F`       | 0      | Floor (passable)              |
| `B`       | 1      | Block / internal obstacle     |
| `M`       | 2      | Monster                       |
| `P`       | 3      | Player spawn / puzzle element |
| `O`       | 4      | Obstacle                      |
| `I`       | 5      | Collectible item              |
| `D`       | 6      | Door                          |
| `S`       | 7      | Secret / special tile / exit  |
| `W`       | 8      | Outer wall                    |
| `-`       | 9      | Void / external padding       |

## Local Setup & Usage
### 1. Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Add raw level files:
Place full-level *.txt* maps (such as *tloz1_1.txt*) into *data/raw/*.

### 3. Run the pipeline via main.py:
- **Prepare dataset** (slice levels into 16x11 rooms, filter duplicates, sample 100 rooms, and augment):
```bash
python main.py --step prepare_data
```
- **Train a model** (base or rules):
```bash
python main.py --step train --mode rules
python main.py --step train --mode base
```

- **Generate a single room with specific door flags** (TOP BOTTOM LEFT RIGHT):
```bash
python main.py --step gen_room --mode rules --doors 1 1 0 1
```
- **Generate a complete multi-room dungeon** (BFS layout + room pool assembly):
```bash
python main.py --step gen_level --mode rules
```
## Running in Google Colab
1. Upload the cvae-rule-loss/ folder to your Google Drive.
2. Open notebooks/colab_runner.ipynb in Google Colab and run the three setup cells:

``` Python
from google.colab import drive
drive.mount('/content/drive')

%cd /content/drive/MyDrive/cvae-rule
!pip install -q -r requirements.txt
```

3. Execute any pipeline stage using shell commands:
``` Python
!python main.py --step prepare_data
!python main.py --step train --mode rules
%run main.py --step gen_level --mode rules
```
