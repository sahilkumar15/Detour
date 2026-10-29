from pathlib import Path

def main():
    output = Path("results/example.txt")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("Detour experiment completed\n")
    print(f"PASS: Saved {output}")

if __name__ == "__main__":
    main()
