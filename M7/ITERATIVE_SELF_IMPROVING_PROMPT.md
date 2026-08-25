Self-improving iterative learning loop for MEGA7

1. Start with the current CSV dataset.
2. Split the last 20 draws into a dedicated holdout file and remove them from the active training CSV.
3. Run the full prediction pipeline with python run_all.py and capture all generated outputs.
4. Compare the predicted ticket and candidate pool against the actual holdout draw.
5. Record the results in a structured log, including:
   - predicted ticket
   - predicted candidate pool
   - actual draw
   - ticket hits
   - pool hits
   - rank percentile
   - success/failure flag
6. Adjust the adaptive model weights based on recent performance.
7. Append the verified holdout draw back into the active training CSV.
8. Repeat the loop for the next draw until all holdout results are processed.
9. Keep improving the system by learning from each verified iteration.
