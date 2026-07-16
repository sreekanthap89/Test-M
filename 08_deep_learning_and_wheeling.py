import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.neural_network import MLPClassifier
from utils import get_run_folder

def load_data(filepath: str) -> pd.DataFrame:
    """Load and prepare the data."""
    df = pd.read_csv(filepath, skipfooter=1, engine="python")
    df["Date"] = pd.to_datetime(df["Date"], format="mixed", dayfirst=True)
    # Parse the winning numbers into a list of ints
    num_cols = ["Winning Number 1", "2", "3", "4", "5", "6"]
    df["numbers"] = df[num_cols].astype(int).values.tolist()
    # Sort chronologically
    df = df.sort_values("Date").reset_index(drop=True)
    # Filter out footer rows
    df = df[df["Date"] >= "2020-01-01"].copy()
    return df

def prepare_ai_data(df: pd.DataFrame, lookback: int = 3, pool_size: int = 40):
    """
    Format data for the Neural Network.
    X: Flattened one-hot vectors of the previous `lookback` draws.
    Y: One-hot vector of the current draw (multi-label classification).
    """
    X, Y = [], []
    
    # Convert each draw into a 40-element binary array (1 if drawn, 0 if not)
    binary_draws = []
    for nums in df["numbers"]:
        b = np.zeros(pool_size)
        for n in nums:
            b[n - 1] = 1
        binary_draws.append(b)
        
    binary_draws = np.array(binary_draws)
    
    # Build time-series windows
    for i in range(lookback, len(binary_draws)):
        # X is the concatenation of the previous 'lookback' draws
        x_window = binary_draws[i - lookback : i].flatten()
        y_target = binary_draws[i]
        X.append(x_window)
        Y.append(y_target)
        
    return np.array(X), np.array(Y), binary_draws

def generate_covering_wheel(pool: list[int], ticket_size: int = 6, match_guarantee: int = 3):
    """
    Greedy Set Cover algorithm to generate a mathematical wheeling system.
    Returns the minimum set of tickets to guarantee `match_guarantee` 
    if `match_guarantee` winning numbers are in the `pool`.
    """
    print(f"\n[Wheeling] Generating a {match_guarantee}-if-{match_guarantee} wheel for {len(pool)} numbers...")
    
    # 1. All possible tickets we could buy (C(12, 6) = 924)
    all_tickets = list(itertools.combinations(pool, ticket_size))
    
    # 2. All required matches we must cover (C(12, 3) = 220)
    requirements = set(itertools.combinations(pool, match_guarantee))
    
    chosen_tickets = []
    
    # Precompute which requirements each ticket covers to speed up
    # A ticket covers a requirement if the requirement is a subset of the ticket
    ticket_covers = {
        ticket: set(itertools.combinations(ticket, match_guarantee))
        for ticket in all_tickets
    }
    
    # Greedy selection loop
    while requirements:
        # Find the ticket that covers the most UNCOVERED requirements
        best_ticket = None
        best_cover_count = -1
        best_covered_set = set()
        
        for ticket, covers in ticket_covers.items():
            # Intersection of what this ticket covers and what's left to cover
            currently_covers = covers.intersection(requirements)
            if len(currently_covers) > best_cover_count:
                best_cover_count = len(currently_covers)
                best_ticket = ticket
                best_covered_set = currently_covers
                
        # Add the best ticket to our wheel
        chosen_tickets.append(best_ticket)
        # Remove the covered requirements from the master list
        requirements -= best_covered_set
        
    print(f"[Wheeling] Wheel generation complete. Guaranteed in {len(chosen_tickets)} tickets.")
    return chosen_tickets

def main():
    print("============================================================")
    print("  STEP 8 — DEEP LEARNING A.I. & WHEELING SYSTEM")
    print("============================================================")
    
    # 1. Load data
    df = load_data("Emirates_Draw_EASY6.csv")
    POOL = 40
    LOOKBACK = 3
    
    print(f"Preparing time-series data (Lookback = {LOOKBACK} draws)...")
    X, Y, binary_draws = prepare_ai_data(df, lookback=LOOKBACK, pool_size=POOL)
    
    # 2. Train the Neural Network
    print("\nTraining Multi-Layer Perceptron (MLP) Neural Network...")
    print("Architecture: Input Layer (120 nodes) -> Hidden (100, 50) -> Output (40 nodes)")
    
    model = MLPClassifier(
        hidden_layer_sizes=(100, 50),
        activation='relu',
        solver='adam',
        max_iter=1000,
        random_state=42
    )
    
    model.fit(X, Y)
    print(f"Model training complete. (Score on training data: {model.score(X, Y):.4f})")
    print("Note: Perfect accuracy on lottery data usually means extreme overfitting!")
    
    # 3. Predict the next draw
    # The input for the next draw is the LAST `LOOKBACK` draws from the dataset
    latest_x = binary_draws[-LOOKBACK:].flatten().reshape(1, -1)
    
    # Get probabilities for each of the 40 numbers
    # MLPClassifier with multi-label output returns an array of shape (n_samples, n_classes)
    # We want the probabilities for our single sample
    probas = model.predict_proba(latest_x)
    next_draw_probs = probas[0]
    
    # Get top 12 numbers
    top_12_indices = np.argsort(next_draw_probs)[::-1][:12]
    top_12_numbers = sorted((top_12_indices + 1).tolist())
    
    print("\n============================================================")
    print("  PHASE 1: A.I. PREDICTION")
    print("============================================================")
    print(f"A.I. Top 12 Predicted Numbers:\n  {top_12_numbers}")
    
    # 4. Combinatorial Wheeling System
    print("\n============================================================")
    print("  PHASE 2: COMBINATORIAL WHEELING")
    print("============================================================")
    print("Buying all possible combinations of these 12 numbers would require 924 tickets.")
    print("Instead, we use a Covering Design (Wheeling).")
    
    # Generate the wheel (Match 3 if 3)
    tickets = generate_covering_wheel(top_12_numbers, ticket_size=6, match_guarantee=3)
    
    print("\nYOUR WHEELED TICKETS:")
    for i, t in enumerate(tickets, 1):
        print(f"  Ticket {i:2d}: {list(t)}")
        
    print(f"\nMATHEMATICAL GUARANTEE: If exactly 3 of the winning numbers fall")
    print(f"anywhere inside {top_12_numbers}, you are mathematically GUARANTEED")
    print(f"to have at least one ticket matching 3 numbers.")
    
    # 5. Visualise AI probabilities
    run_dir = get_run_folder()
    
    plt.figure(figsize=(14, 6))
    bars = plt.bar(range(1, POOL + 1), next_draw_probs, color='#95a5a6')
    
    # Highlight top 12
    for idx in top_12_indices:
        bars[idx].set_color('#e74c3c')
        
    plt.title("A.I. Neural Network Probability Prediction for Next Draw", fontsize=14, pad=15)
    plt.xlabel("Number", fontsize=12)
    plt.ylabel("Predicted Probability", fontsize=12)
    plt.xlim(0.5, POOL + 0.5)
    plt.xticks(range(1, POOL + 1))
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', label='Top 12 AI Picks (Wheeled)'),
        Patch(facecolor='#95a5a6', label='Other Numbers')
    ]
    plt.legend(handles=legend_elements)
    
    plt.tight_layout()
    chart_path = f"{run_dir}/step8_deep_learning_wheel.png"
    plt.savefig(chart_path, dpi=120)
    plt.close()
    
    print(f"\n[OK] Chart saved -> {chart_path}")
    
if __name__ == "__main__":
    main()
